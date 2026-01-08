# Smart Manufacturing - Industrial Predictive Maintenance Platform
---
Project: Smart Manufacturing - Industrial Predictive Maintenance Platform

Author: CHOU CHIA-HSUAN, Fu Ji-Ru, Wang Pei-Ling

Date: 2026-01-02

Course: Statistical Consulting

---

## 1. Web Application (Streamlit & Cloud Run)

### Functions
1. Provides an interactive interface for inputting parameters
2. Performs real-time machine failure risk prediction
3. Visualizes model explanations through global and individual SHAP plots


### First Page: Parameter Input

Users can input machine parameters to perform failure risk prediction.  
Each variable can be adjusted using a **slider** or by **manually typing a numeric value**.

1. If a variable value is unavailable, users may select **“I don’t know”**
2. If required inputs are left empty, the system will display a warning message before prediction.

After completing the inputs, click **“View my prediction”** to proceed to the results page.

![Demo1](img/page1_1.png)
![Demo2](img/page1_2.png)
![Demo3](img/page1_3.png)

### Second Page: Prediction Results & Model Explainability

#### 1. Failure Probability Overview
The system displays the **overall machine failure probability** along with a clear risk level indicator (e.g., **HIGH RISK**).  

![Demo4](img/page2_1.png)

#### 2. Individual Explanation (Local SHAP)
**2-1 Waterfall plot**
![Demo5](img/page2_2.png)

**2-2 Force plot**
![Demo6](img/page2_3.png)

#### 3. Global Explanation (Global SHAP)
**3-1 Beeswarm**
![Demo7](img/page2_4.png)

**3-2 Feature Importance Bar plot**
![Demo8](img/page2_5.png)


## 2. Code Structure (Cloud Deployment)

### Cloud Run Deployment Files

<pre> 
├── firstpage.py                  # Streamlit web application
├── ai4i_xgb_pipeline_final.pkl   # Trained XGBoost pipeline (model + preprocessing)
├── Dockerfile                    # Docker build configuration
├── .dockerignore                 # Files excluded from Docker image
</pre>        

### Local
<pre> 
├── ai4i2020.csv        # AI4I 2020 dataset
├── pipeline.ipynb      # Model training and evaluation notebook
├── prompt.txt          # Command-line instructions for cloud deployment
├── README.md        
</pre>  


## 3. Dataset

**AI4I 2020 Predictive Maintenance Dataset**

- **Total samples**: 10,000 machine operation logs  
- **Target variable**: Machine failure(1: Failure, 0: Normal) 
- **Class imbalance**: Failure events are rare (approximately 3.4%)

### Original Features
- Type: Machine quality variant (L / M / H)
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

### Engineered Features
- **Temp Diff** = Process temperature − Air temperature  
- **Power** = Torque × Rotational speed  

## 4. Model Experiments & Selection

### 4.1 Data Split
- Training set: 8,000 samples  
- Test set: 2,000 samples  
- Stratified split

### 4.2 Model Comparison

| Model | Accuracy | Precision | Recall  | F1-score  |
|------|---------|---------------------|------------------|-------------------|
| Logistic Regression | 87% | 0.19 | 0.81 | 0.30 |
| SVM (RBF Kernel) | 92% | 0.27 | 0.81 | 0.40 |
| Random Forest | 97% | 0.51 | 0.72 | 0.59 |
| **XGBoost (Selected)** | **98.95%** | **0.87** | **0.81** | **0.84** |


## 5. How to set up to cloud

![Structure](img/structure.png)

The system is deployed using **Docker** and **Google Cloud Run** to ensure a consistent execution environment and scalable access.

The deployment process includes:
1. Containerizing the Streamlit application and trained XGBoost model using Docker  
2. Uploading the Docker image to Google Artifact Registry  
3. Deploying the container to Google Cloud Run with predefined CPU, memory, and concurrency settings  
4. Automatically generating an HTTPS for public access  

