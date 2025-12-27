# Radcom Traffic Classification System 🚀

This project is a high-performance, scalable machine learning system designed to classify encrypted network traffic.
---

## 🧠 Machine Learning Approach

### Algorithm
The system utilizes the **Random Forest Classifier** for both tasks:

- **App Classification**:  
  Trained with **200 estimators** and a **maximum depth of 25** to handle the high complexity of 128 target classes.

- **Attribution Classification**:  
  Trained with **100 estimators**, using all available CPU cores (`n_jobs=-1`) for efficient processing.

### Preprocessing & Feature Engineering
To ensure the model generalizes across different networks and avoids overfitting on identity data, the following steps are performed:

- **Identity Removal**:  
  Columns such as `Source_IP`, `Destination_IP`, and `Timestamp` are dropped to focus purely on behavioral patterns.

- **Protocol Encoding**:  
  Network protocols are encoded numerically:
  - TCP → `6`
  - UDP → `17`

- **Feature Alignment**:  
  During inference, incoming data is reindexed to match the exact feature order used during training, with missing values filled with zeros.

---

## 🏗️ System Architecture

The project follows a **decoupled microservices architecture** designed to handle high-traffic *hexabyte-scale* workloads through horizontal scaling.

- **FastAPI**  
  REST API gateway serving as the entry point for inference requests using the OpenAPI standard.

- **Celery & Redis**  
  Asynchronous task queue that prevents API blocking during heavy ML computations, enabling massive scale.

- **Streamlit**  
  User-friendly web interface for file uploads, real-time status polling, and result visualization.

- **Docker**  
  All components are containerized to ensure environment consistency across development and production.

---

## 🚀 Getting Started

### 1. Model Training & Testing

> ⚠️ Models are **not included** in the source submission.  
> You must generate them before starting the Docker services.

The trained models will be saved under the `models/` directory.

#### Train the models
```bash
# Generate the Application Model
python3 src/train_app.py

# Generate the Attribution Model
python3 src/train_att.py
```

#### Test the models
Evaluate accuracy and view classification reports:
```bash
python3 src/test_models.py
```

---

### 2. Activating the Docker Environment

Ensure your `models/` directory contains the generated `.pkl` files before launching the containers.

```bash
# Build and start all services
sudo docker compose up --build
```

---

### 3. Accessing the System

Once the containers are running:

- **Streamlit Web UI**  
  http://localhost:8501  
  Upload your validation CSV and receive results with the prediction column.

- **FastAPI Docs (OpenAPI)**  
  http://localhost:8000/docs  
  Interact directly with the REST API endpoints.

---

## 📂 Project Structure

```
├── Dockerfile.app              # FastAPI inference service container
├── Dockerfile.worker           # Celery worker container for background ML tasks
├── README.md                   # Project documentation
├── docker-compose.yml          # Full stack orchestration (API, Worker, Redis, UI)
├── requirements.txt            # Python dependencies
│
├── data                        # Training, testing, and validation datasets
│   ├── APP-1                   # Application classification datasets
│   │   ├── radcom_app_train.csv
│   │   ├── radcom_app_test.csv
│   │   └── radcom_app_val_without_labels.csv
│   └── attribution             # Traffic attribution datasets
│       ├── radcom_att_train.csv
│       ├── radcom_att_test.csv
│       └── radcom__att_val_without_labels.csv
│
├── models                      # Trained models and feature definitions
│   ├── model_app.pkl           # Random Forest model for application classification
│   ├── model_att.pkl           # Random Forest model for attribution classification
│   ├── app_features.pkl        # Ordered feature list for app inference
│   └── att_features.pkl        # Ordered feature list for attribution inference
│
├── predictions                 # Output prediction results
│   ├── app_predictions.csv
│   └── att_predictions.csv
│
├── src                         # Core application source code
│   ├── api.py                  # FastAPI server and REST endpoints
│   ├── tasks.py                # Celery task definitions for asynchronous inference
│   ├── worker.py               # Celery worker initialization and model loading
│   ├── streamlit_app.py        # Streamlit web UI for file upload and result visualization
│   └── utils.py                # Shared preprocessing and feature-engineering utilities
│
├── train                       # Model training scripts
│   ├── train_app.py            # Application classifier training pipeline
│   └── train_att.py            # Attribution classifier training pipeline
│
└── test_models.py              # Model evaluation and validation reports
```
