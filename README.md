# Customer Churn Prediction - Production MLOps Pipeline

> A full end-to-end MLOps system built to demonstrate production-ready machine learning engineering practices - experiment tracking, model serving, containerization, CI/CD, and monitoring.

---

## 🏗️ Architecture

<img width="1110" height="1176" alt="image" src="https://github.com/user-attachments/assets/94deb145-35a9-428a-99df-df63dc0971b7" />


---

## 🚀 MLOps Layers

| Layer | Technology | Description |
|-------|-----------|-------------|
| **Experiment Tracking** | MLflow | Log hyperparameters, metrics, artifacts for 5 models |
| **Model Serving** | FastAPI | REST API with `/predict` and `/batch_predict` endpoints |
| **Containerization** | Docker | Dockerized API with docker-compose |
| **CI/CD** | GitHub Actions | Auto test → build → deploy on every push |
| **Monitoring** | Evidently AI + Streamlit | Data drift detection + live dashboard |
| **Reproducibility** | MLproject | One-command pipeline reproduction |

---

## 📊 Model Performance Results

| Model | AUC | F1 | Accuracy |
|-------|-----|----|----------|
| **RandomForest ⭐** | **0.8448** | 0.5452 | 79.99% |
| LogisticRegression | 0.8407 | 0.5945 | 80.06% |
| XGBoost | 0.8413 | 0.5744 | 79.70% |
| XGBoost-Tuned | 0.8366 | 0.5956 | 80.34% |
| LightGBM | 0.8374 | 0.5839 | 79.77% |

> Best model: **RandomForest** (AUC = 0.8448) registered in MLflow Model Registry

---

## 📁 Project Structure

```
churn-mlops-pipeline/
├── data/
│   ├── raw/                    # Raw IBM Telco dataset
│   └── processed/              # Cleaned train/test splits + feature order
├── src/
│   ├── data_processing.py      # Data cleaning, encoding, splitting
│   ├── feature_engineering.py  # Feature engineering utilities
│   ├── train.py                # Train 5 models, log to MLflow
│   ├── evaluate.py             # Model evaluation
│   └── predict.py              # Prediction utilities
├── api/
│   ├── main.py                 # FastAPI app with /predict & /batch_predict
│   └── schemas.py              # Pydantic input/output schemas
├── monitoring/
│   ├── drift_detection.py      # Evidently AI drift detection
│   └── dashboard.py            # Streamlit monitoring dashboard
├── tests/
│   ├── test_api.py             # API endpoint tests
│   └── test_model.py           # Model tests
├── .github/
│   └── workflows/
│       └── ci_cd.yml           # GitHub Actions CI/CD pipeline
├── Dockerfile                  # API containerization
├── docker-compose.yml          # Multi-service orchestration
├── MLproject                   # MLflow reproducibility file
├── python_env.yaml             # Python environment spec
└── requirements.txt            # Pinned dependencies
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/1825Vaishnavi/churn-mlops-pipeline.git
cd churn-mlops-pipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
```bash
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
     -o data/raw/Telco-Customer-Churn.csv
```

### 4. Run the full pipeline
```bash
python src/data_processing.py
python src/train.py
```

### 5. Start the API
```bash
python -m uvicorn api.main:app --reload
# Visit http://127.0.0.1:8000/docs
```

### 6. View MLflow experiments
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
# Visit http://127.0.0.1:5000
```

### 7. Run monitoring dashboard
```bash
python monitoring/drift_detection.py
python -m streamlit run monitoring/dashboard.py
# Visit http://localhost:8501
```

### 8. Run with Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## 🔌 API Endpoints

### `POST /predict` - Single prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 0,
    "Contract": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 2
  }'
```

**Response:**
```json
{
  "churn_probability": 0.3814,
  "churn_prediction": false,
  "risk_level": "Low"
}
```

### `GET /health` - Health check
```json
{"status": "healthy", "model_loaded": true}
```

---

## 🔄 CI/CD Pipeline

Every push to `main` triggers:

```
Push to main
     │
     ▼
┌─────────────┐
│   test job   │
│ 1. Install   │
│ 2. Download  │
│    dataset   │
│ 3. Process   │
│    data      │
│ 4. Train     │
│    model     │
│ 5. Run tests │
└──────┬──────┘
       │  pass
       ▼
┌─────────────┐
│ docker job   │
│ 1. Build     │
│    image     │
└─────────────┘
```

---

## 📈 Monitoring Dashboard

The Streamlit dashboard provides 4 views:

- **Model Performance** - AUC/F1/Accuracy comparison across all models
- **Data Drift** - Evidently AI drift scores per feature
- **Prediction Analytics** - Risk distribution, churn probability histogram, live predictions
- **System Health**- API/MLflow status, CI/CD pipeline status, response times

---

## 🔁 Reproduce Training

Using MLflow Projects:
```bash
mlflow run . -e train
```

Or run individual steps:
```bash
mlflow run . -e data_processing
mlflow run . -e train
```

---

## 📚 Dataset

**IBM Telco Customer Churn Dataset**
- 7,043 customers
- 21 features (demographics, services, charges)
- 26.54% churn rate
- Source: [IBM Sample Data](https://github.com/IBM/telco-customer-churn-on-icp4d)

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11 |
| ML | scikit-learn, XGBoost, LightGBM |
| Tracking | MLflow |
| API | FastAPI, Uvicorn |
| Validation | Pydantic v2 |
| Container | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI, Streamlit |
| Database | SQLite (MLflow backend) |

---

## 👩‍💻 Author

**Vaishnavi Mallikarjun**
Master's in Data Analytics Engineering - Northeastern University

[GitHub](https://github.com/1825Vaishnavi) | [LinkedIn](https://linkedin.com/in/vaishnavi)
