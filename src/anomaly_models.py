# src/anomaly_models.py

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    IF_MODEL_DIR,
    AE_MODEL_DIR,
    RANDOM_STATE,
    ANOMALY_QUANTILE,
)


# ---------- Utility ----------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize a 1D score array to [0, 1]."""
    scores = np.asarray(scores, dtype=float)
    min_val = float(scores.min())
    max_val = float(scores.max())
    if max_val == min_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


# ---------- PyTorch Autoencoder ----------

class WindowAutoencoder(nn.Module):
   
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def _train_autoencoder_torch(
    X_win_scaled: np.ndarray,
    input_dim: int,
    epochs: int = 20,
    batch_size: int = 256,
) -> Tuple[WindowAutoencoder, np.ndarray]:
    """
    Train a PyTorch autoencoder on scaled window data.

    Returns:
        model: trained WindowAutoencoder
        recon_errors: numpy array of reconstruction errors per sample
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X_tensor = torch.from_numpy(X_win_scaled.astype(np.float32))
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = WindowAutoencoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_in, batch_target in loader:
            batch_in = batch_in.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            out = model(batch_in)
            loss = criterion(out, batch_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_in.size(0)

        epoch_loss /= len(dataset)
        print(f"[AE train] Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f}")

    # Compute reconstruction errors for entire dataset
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        recon = model(X_tensor)
        errors = torch.mean((X_tensor - recon) ** 2, dim=1)
        recon_errors = errors.cpu().numpy()

    return model, recon_errors


# ---------- Training ----------

def train_anomaly_models(
    X_windows: np.ndarray,
    X_features: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    save: bool = True,
) -> Dict:
    """
    Train:
      - Isolation Forest on window-level engineered features (X_features)
      - PyTorch Autoencoder on flattened window sequences (X_windows)

    Returns a dict with models, scalers, thresholds and training scores.

    All thresholds (for IF, AE, hybrid) are computed using ANOMALY_QUANTILE.
    """
    X_windows = np.asarray(X_windows, dtype=float)
    X_features = np.asarray(X_features, dtype=float)

    n_windows, window_dim = X_windows.shape
    _, feat_dim = X_features.shape

    print(f"[train] X_windows shape = {X_windows.shape}, X_features shape = {X_features.shape}")

    # ----- 1) Isolation Forest on engineered features -----
    print("[train] Scaling features for Isolation Forest...")
    if_scaler = StandardScaler()
    X_feat_scaled = if_scaler.fit_transform(X_features)

    print("[train] Training Isolation Forest...")
    if_model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if_model.fit(X_feat_scaled)

    # IF anomaly scores: lower = more abnormal, so invert
    if_raw_scores = -if_model.score_samples(X_feat_scaled)
    if_scores_norm = _normalize_scores(if_raw_scores)

    # ----- 2) Autoencoder on window sequences -----
    print("[train] Scaling window sequences for Autoencoder...")
    ae_scaler = StandardScaler()
    # Replace inf with nan, then drop rows with nan before AE
    X_windows = np.where(np.isfinite(X_windows), X_windows, np.nan)

    # Drop any windows that contain NaN
    mask_valid = ~np.isnan(X_windows).any(axis=1)
    X_windows = X_windows[mask_valid]
    X_features = X_features[mask_valid]  # keep alignment
    print(f"[clean] AE training: removed {(~mask_valid).sum()} windows with NaN/inf")

    X_win_scaled = ae_scaler.fit_transform(X_windows)

    print("[train] Building & training PyTorch Autoencoder...")
    ae_model, ae_raw_errors = _train_autoencoder_torch(
        X_win_scaled=X_win_scaled,
        input_dim=window_dim,
        epochs=epochs,
        batch_size=batch_size,
    )
    ae_scores_norm = _normalize_scores(ae_raw_errors)

    # ----- 3) Hybrid score -----
    hybrid_scores = 0.5 * (if_scores_norm + ae_scores_norm)

    # ----- 4) Data-driven thresholds -----
    if_thresh = float(np.quantile(if_scores_norm, ANOMALY_QUANTILE))
    ae_thresh = float(np.quantile(ae_scores_norm, ANOMALY_QUANTILE))
    hybrid_thresh = float(np.quantile(hybrid_scores, ANOMALY_QUANTILE))

    thresholds = {
        "if_threshold": if_thresh,
        "ae_threshold": ae_thresh,
        "hybrid_threshold": hybrid_thresh,
        "quantile": ANOMALY_QUANTILE,
        "ae_input_dim": window_dim,  # needed when loading model back
    }

    print("[train] Thresholds (quantile", ANOMALY_QUANTILE, "):")
    print("       IF:", if_thresh, " AE:", ae_thresh, " Hybrid:", hybrid_thresh)

        # ----- 5) Build training scores dataframe with anomaly flags -----
    if_is_anom = if_scores_norm >= if_thresh
    # Handle potential NaNs in AE/hybrid gracefully: treat them as non-anomalous
    ae_is_anom = np.where(np.isfinite(ae_scores_norm), ae_scores_norm >= ae_thresh, False)
    hybrid_is_anom = np.where(np.isfinite(hybrid_scores), hybrid_scores >= hybrid_thresh, False)

    training_scores_df = pd.DataFrame(
        {
            "if_score": if_scores_norm,
            "ae_score": ae_scores_norm,
            "hybrid_score": hybrid_scores,
            "if_is_anomaly": if_is_anom,
            "ae_is_anomaly": ae_is_anom,
            "hybrid_is_anomaly": hybrid_is_anom,
        }
    )

    # ----- 5) Save models + scalers + thresholds -----
    if save:
        _ensure_dir(Path(IF_MODEL_DIR))
        _ensure_dir(Path(AE_MODEL_DIR))

        # Isolation Forest & its scaler
        joblib.dump(if_model, Path(IF_MODEL_DIR) / "isolation_forest_model.joblib")
        joblib.dump(if_scaler, Path(IF_MODEL_DIR) / "if_scaler.joblib")

        # Autoencoder & its scaler
        torch.save(ae_model.state_dict(), Path(AE_MODEL_DIR) / "autoencoder_model.pt")
        joblib.dump(ae_scaler, Path(AE_MODEL_DIR) / "ae_scaler.joblib")

        # Common thresholds
        joblib.dump(
            thresholds,
            Path(IF_MODEL_DIR) / "anomaly_thresholds.joblib",
        )

        print(f"[train] Models and thresholds saved under:\n  {IF_MODEL_DIR}\n  {AE_MODEL_DIR}")

      # ----- 5) Build training scores dataframe with anomaly flags -----
    if_is_anom = if_scores_norm >= if_thresh

    # AE/hybrid can produce NaN if AE loss was NaN — handle safely
    ae_is_anom = np.where(
        np.isfinite(ae_scores_norm),
        ae_scores_norm >= ae_thresh,
        False
    )
    hybrid_is_anom = np.where(
        np.isfinite(hybrid_scores),
        hybrid_scores >= hybrid_thresh,
        False
    )

    training_scores_df = pd.DataFrame(
        {
            "if_score": if_scores_norm,
            "ae_score": ae_scores_norm,
            "hybrid_score": hybrid_scores,
            "if_is_anomaly": if_is_anom,
            "ae_is_anomaly": ae_is_anom,
            "hybrid_is_anomaly": hybrid_is_anom,
        }
    )

    return {
        "if_model": if_model,
        "if_scaler": if_scaler,
        "ae_model": ae_model,
        "ae_scaler": ae_scaler,
        "thresholds": thresholds,
        "scores_df": training_scores_df,
    }


# ---------- Loading ----------

def load_anomaly_models() -> Dict:
    """
    Load Isolation Forest, PyTorch Autoencoder, their scalers, and thresholds from disk.
    """
    if_dir = Path(IF_MODEL_DIR)
    ae_dir = Path(AE_MODEL_DIR)

    if_model = joblib.load(if_dir / "isolation_forest_model.joblib")
    if_scaler = joblib.load(if_dir / "if_scaler.joblib")
    thresholds = joblib.load(if_dir / "anomaly_thresholds.joblib")

    ae_scaler = joblib.load(ae_dir / "ae_scaler.joblib")

    # Rebuild autoencoder with correct input dimension
    if "ae_input_dim" in thresholds:
        input_dim = int(thresholds["ae_input_dim"])
    else:
        # fallback: infer from scaler
        input_dim = ae_scaler.mean_.shape[0]

    ae_model = WindowAutoencoder(input_dim=input_dim)
    state_dict = torch.load(ae_dir / "autoencoder_model.pt", map_location="cpu")
    ae_model.load_state_dict(state_dict)
    ae_model.eval()

    return {
        "if_model": if_model,
        "if_scaler": if_scaler,
        "ae_model": ae_model,
        "ae_scaler": ae_scaler,
        "thresholds": thresholds,
    }


# ---------- Scoring for new data ----------

def compute_anomaly_scores_for_windows(
    X_windows: np.ndarray,
    X_features: np.ndarray,
    models: Dict,
) -> pd.DataFrame:
    """
    Use trained models to compute IF, AE and hybrid anomaly scores
    for each window.

    Returns:
        DataFrame with columns:
            if_score, ae_score, hybrid_score,
            if_is_anomaly, ae_is_anomaly, hybrid_is_anomaly
    """
    X_windows = np.asarray(X_windows, dtype=float)
    X_features = np.asarray(X_features, dtype=float)

    if_model = models["if_model"]
    if_scaler = models["if_scaler"]
    ae_model = models["ae_model"]
    ae_scaler = models["ae_scaler"]
    thresholds = models["thresholds"]

    # ----- Isolation Forest scores -----
    X_feat_scaled = if_scaler.transform(X_features)
    if_raw_scores = -if_model.score_samples(X_feat_scaled)
    if_scores_norm = _normalize_scores(if_raw_scores)

    # ----- Autoencoder scores (PyTorch) -----
    X_win_scaled = ae_scaler.transform(X_windows)
    X_tensor = torch.from_numpy(X_win_scaled.astype(np.float32))
    ae_model.eval()
    with torch.no_grad():
        recon = ae_model(X_tensor)
        errors = torch.mean((X_tensor - recon) ** 2, dim=1)
        ae_raw_errors = errors.cpu().numpy()
    ae_scores_norm = _normalize_scores(ae_raw_errors)

    # ----- Hybrid -----
    hybrid_scores = 0.5 * (if_scores_norm + ae_scores_norm)

    # ----- Labels based on stored thresholds -----
    if_thr = thresholds["if_threshold"]
    ae_thr = thresholds["ae_threshold"]
    hy_thr = thresholds["hybrid_threshold"]

    if_is_anom = if_scores_norm >= if_thr
    ae_is_anom = ae_scores_norm >= ae_thr
    hy_is_anom = hybrid_scores >= hy_thr

    return pd.DataFrame(
        {
            "if_score": if_scores_norm,
            "ae_score": ae_scores_norm,
            "hybrid_score": hybrid_scores,
            "if_is_anomaly": if_is_anom,
            "ae_is_anomaly": ae_is_anom,
            "hybrid_is_anomaly": hy_is_anom,
        }
    )
