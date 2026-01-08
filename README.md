# Smart Meter Theft Detection - Spike & Dip Analysis

## Project Overview
This project provides an end-to-end solution for detecting abnormal power consumption and potential theft-like events using high-frequency smart meter data. The system analyzes 5-second interval data (Power, Voltage, Current, and Power Factor) to identify anomalies through a hybrid machine learning approach.

## Tech Stack
* **Deep Learning**: PyTorch (LSTM Autoencoder for sequence reconstruction)
* **Machine Learning**: Scikit-Learn (Isolation Forest for tabular anomaly detection)
* **Web Framework**: Streamlit (Interactive Data Dashboard)
* **Data Science**: Pandas, NumPy, Matplotlib, Plotly, SciPy
* **Automation**: Python-based pipeline for continuous data processing

## Data Pipeline Architecture
The project follows a structured sequence:
1. **Inspiration**: Loads ~1 million records of 5-second interval data.
2. **Hybrid ML Modeling**: 
    * **Isolation Forest**: Analyzes summary statistics per 10-minute window.
    * **Autoencoder**: Learns complex temporal patterns to flag reconstruction errors.
3. **Theft Cues**: Applies logic for Meter Freeze, Sharp Power Spikes, and PF Abnormalities to identify suspicious activity.

## Dashboard Preview
Below are the visual outputs from the detection system:

### 1. Daily Spike & Theft Timeline
![Spike Timeline](REPLACE_WITH_YOUR_GITHUB_ISSUE_LINK_HERE)

### 2. Parameter Load Patterns (Power, Voltage, Current, PF)
![Load Patterns](REPLACE_WITH_YOUR_GITHUB_ISSUE_LINK_HERE)

### 3. Event Audit & Theft Cues Table
![Event Table](REPLACE_WITH_YOUR_GITHUB_ISSUE_LINK_HERE)

## Installation and Setup
1. Create a virtual environment:
   `python -m venv .venv`
2. Activate the environment:
   `.\.venv\Scripts\Activate.ps1`
3. Install required dependencies:
   `pip install -r requirements.txt`

## Execution Guide
### Step 1: Run the Data Pipeline
Execute the core logic to process raw data and generate ML scores:
`python main_run_pipeline.py`

### Step 2: Launch the Dashboard
Start the visualization tool to audit detected theft events:
`streamlit run dashboard/app.py`
