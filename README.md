# Smart Meter Theft Detection - Spike & Dip Analysis

## Project Overview
This project provides an end-to-end solution for detecting abnormal power consumption and potential theft-like events using high-frequency smart meter data. The system analyzes 5-second interval data (Power, Voltage, Current, and Power Factor) to automatically:
* Detect abnormal consumption patterns using a hybrid AI approach.
* Group anomalous 10-minute windows into longer, actionable events.
* Apply domain-specific "theft cues" to identify suspicious tampering or meter bypass.
* Provide an interactive dashboard for daily auditing and validation against raw data.

## Tech Stack
* **Deep Learning**: PyTorch (Dense Autoencoder for pattern reconstruction)
* **Machine Learning**: Scikit-Learn (Isolation Forest for window-level anomaly detection)
* **Dashboard**: Streamlit (Web interface and visualization)
* **Data Processing**: Pandas, NumPy, SciPy
* **Visualization**: Plotly, Matplotlib
* **Environment**: Python 3.11+

## Data Pipeline Architecture
The project follows a structured sequence from raw data to final audit:
1. **Data Ingestion**: Processes ~1 million records of 5-second interval data from `data/raw/house_data.csv`.
2. **Feature Engineering**: Computes step differences and 10-minute sliding window statistics (Mean, Std, Max Spikes, Flatness).
3. **Hybrid ML Scoring**: Combines Isolation Forest scores with Autoencoder reconstruction errors for high-accuracy anomaly detection.
4. **Event Detection**: Groups consecutive anomalous windows into continuous events (e.g., 40–60 minute periods).
5. **Theft Rules Engine**: Applies cues like Meter Freeze, Sharp Power Spikes, and PF Abnormality to mark events as "Theft-like".

## Dashboard Showcase
Below are the visual outputs from the detection system:

### 1. Main Overview & Key Metrics
![Dashboard Overview](https://github.com/ashrith-sambarju/THEFT-DETECTION---SPIKE-DIP/issues/1#issue-3792355882)

### 2. Theft Detection Timeline (Hybrid Anomaly Score)
![Anomaly Timeline](https://github.com/ashrith-sambarju/THEFT-DETECTION---SPIKE-DIP/issues/2)

### 3. Parameter Load Pattern (Normalized Day View)
![Load Pattern](https://github.com/ashrith-sambarju/THEFT-DETECTION---SPIKE-DIP/issues/3)

### 4. Event Audit & Theft Cues Table
![Event Details](PASTE_LINK_4_HERE)

## Installation and Setup
1. **Create Virtual Environment**:
   `python -m venv .venv`
2. **Activate Environment**:
   `.\.venv\Scripts\Activate.ps1`
3. **Install Dependencies**:
   `pip install -r requirements.txt`

## Execution Guide
### Step 1: Run the Data Pipeline
Execute the core logic to process raw data and generate ML scores:
`python main_run_pipeline.py`

### Step 2: Launch the Dashboard
Start the visualization tool to audit detected theft events:
`streamlit run dashboard/app.py`
