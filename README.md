# ⚙️ Predictive Maintenance System (IoT + AI)

A complete **Predictive Maintenance** project that uses **simulated IoT sensor data**, **machine learning (Random Forest)**, and **deep learning (LSTM)** to predict potential equipment failures *before they happen*.  
Includes a **Streamlit dashboard**, **FastAPI service**, and **model performance visualizations**.

---

## 📁 Project Overview

Predictive maintenance helps industries minimize downtime by predicting failures in advance using real-time sensor data such as:

- Temperature  
- Vibration  
- Pressure  
- RPM  

This project simulates multi-sensor data streams for several machines, processes it, trains ML/DL models, and provides:
- 📊 **Data analytics and feature engineering**
- 🤖 **Machine learning and deep learning model training**
- 🧮 **Model comparison (RF vs LSTM)**
- 🌐 **Interactive dashboard using Streamlit**
- ⚡ **REST API (FastAPI) for real-time predictions**

---

## 🧩 Tech Stack

| Category | Tools Used |
|-----------|-------------|
| **Programming Language** | Python 3.10 |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Modeling** | RandomForestClassifier, PyTorch (LSTM) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Deployment** | Streamlit, FastAPI, Uvicorn |
| **Testing** | Pytest |
| **IDE/Environment** | PyCharm / Jupyter Notebook |

---

## 🧱 Project Structure

predictive_maintenance/
│
├── app/
│ ├── streamlit_app.py # Interactive dashboard
│ └── api_service.py # FastAPI service for predictions
│
├── data/
│ ├── raw/ # Raw (simulated) sensor data
│ └── processed/ # Feature-engineered datasets
│
├── notebooks/
│ ├── 01_eda_visualization.ipynb
│ ├── 02_model_training_sklearn.ipynb
│ ├── 03_model_training_pytorch.ipynb
│ └── 04_model_comparison.ipynb
│
├── models/
│ ├── rf_model.pkl
│ ├── lstm_model.pt
│ └── scaler.pkl
│
├── src/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ └── inference.py
│
├── tests/
│ └── test_model_predictions.py
│
├── reports/
│ └── visuals/
│ ├── eda_visuals.png
│ ├── model_performance_comparison.png
│ └── project_architecture.png
│
└── README.md

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/predictive_maintenance.git
cd predictive_maintenance
2️⃣ Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # for Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Generate sample IoT sensor data
python data/sample_data_generator.py

5️⃣ Preprocess and engineer features
python src/data_preprocessing.py

6️⃣ Train models (optional, already saved)

Run Jupyter notebooks inside /notebooks/ for model training and comparison.

7️⃣ Launch the dashboard
streamlit run app/streamlit_app.py


Access the app at 👉 http://localhost:8501

8️⃣ (Optional) Run the API service
uvicorn app.api_service:app --reload --port 8000


Then test at 👉 http://127.0.0.1:8000/docs

📊 Key Results
Model	Accuracy	ROC-AUC
Random Forest	~0.89	0.91
LSTM (Deep Learning)	~0.88	0.87
