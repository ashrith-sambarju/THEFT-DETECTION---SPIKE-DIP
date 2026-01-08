# Smart Meter Theft Detection - Spike & Dip Analysis

## Project Overview
[cite_start]This project provides an end-to-end solution for detecting abnormal power consumption and potential theft-like events using high-frequency smart meter data[cite: 3, 4]. [cite_start]The system analyzes 5-second interval data (Power, Voltage, Current, and Power Factor) to identify anomalies through a hybrid machine learning approach[cite: 13, 14, 74].


## Data Pipeline Architecture
[cite_start]The project follows a structured sequence from raw data ingestion to visualization[cite: 290]:

1.  [cite_start]**Data Ingestion**: Loads and cleans ~1,044,263 records of 5-second interval CSV data from `data/raw/house_data.csv`[cite: 12, 14, 15].
2.  [cite_start]**Feature Engineering**: Computes step differences for each parameter and creates fixed 10-minute sliding windows[cite: 25, 39, 40].
3.  [cite_start]**Hybrid ML Modeling**: Utilizes two unsupervised models to identify anomalies[cite: 73, 74]:
    * [cite_start]**Isolation Forest**: Analyzes engineered window features like power statistics and spike magnitudes[cite: 75, 82].
    * [cite_start]**Autoencoder (PyTorch)**: A dense neural network that learns normal patterns and flags reconstruction errors[cite: 76, 86, 88].
4.  [cite_start]**Anomaly Scoring**: Combines model outputs into a hybrid score, flagging the top 10% of windows as anomalous[cite: 96, 98].
5.  [cite_start]**Event Detection**: Groups consecutive anomalous windows into distinct "Events" to capture long-term abnormal behavior[cite: 128, 134].
6.  [cite_start]**Theft Cues**: Applies domain-specific rules (Meter Freeze, Sharp Power Spikes, PF Abnormality) to events to identify potential tampering[cite: 149, 151, 152].

## Project Structure
* [cite_start]**dashboard/**: Contains `app.py`, the Streamlit visualization application[cite: 189].
* [cite_start]**data/**: Includes `raw/` for input data and `processed/` for pipeline outputs[cite: 12, 23, 105].
* [cite_start]**models/**: Stores trained weights and scalers for the Isolation Forest and Autoencoder[cite: 105].
* [cite_start]**src/**: Core source code for data loading, feature engineering, modeling, and rule application[cite: 15, 26, 47, 78].
* **main_run_pipeline.py**: The primary execution script that runs the entire processing and ML pipeline.

## Installation and Setup
1. Ensure Python 3.11+ is installed.
2. Create a virtual environment:
   `python -m venv .venv`
3. Activate the environment:
   `.\.venv\Scripts\Activate.ps1`
4. Install required dependencies:
   `pip install -r requirements.txt`

## Execution Guide
To run the system, follow these steps in order:

### 1. Run the Data Pipeline
Execute the main script to process data, train models, and generate summary files:
`python main_run_pipeline.py`
[cite_start]This populates the `data/processed/` folder with results such as `events_with_cues.csv` and `spike_timeline.csv`[cite: 168, 187].

### 2. Launch the Dashboard
Once the pipeline has completed, start the interactive Streamlit interface:
`streamlit run dashboard/app.py`

## Dashboard Features
* [cite_start]**Spike Timeline**: Visualization of daily maximum hybrid anomaly scores[cite: 207, 210].
* [cite_start]**Theft Cues Timeline**: Tracking of the number of theft-like events per day[cite: 221, 224].
* [cite_start]**Selected Day Analysis**: Drill down into energy consumption by segment (Night, Morning, Afternoon, Evening) and normalized 4-parameter load patterns[cite: 229, 231, 237].
* [cite_start]**Event Audit Table**: Detailed breakdown of event timings, durations, and the specific theft cues triggered[cite: 250, 261].