### README.md

````markdown
# 2025 Disaster Prediction & Loss Analysis Project

This project aims to predict the frequency of disasters in 2025 and estimate the associated social losses (economic damage and death toll) using a combination of Transformer and XGBoost models.

This is ML&Feature Engineering part.

## 📂 Project Structure

### Directories

- **`dataset/`**: Input directory. **Note:** You must place your two raw database files into this folder before starting.
- **`output/`**: Stores model visualization results and prediction outputs.
- **`output_model/`**: Stores the saved model weights/checkpoints.
- **`processed_data/`**: Stores intermediate data generated after performing feature engineering.

### Scripts & Files

- **`environment.yml`**: Configuration file to construct the virtual environment.
- **`feature_engineering.py`**: script to clean data and extract relevant features.
- **`transformer_prediction.py`**: Uses a Transformer model to predict the _frequency_ of disasters for 2025.
- **`XGBoost.py`**: Trains the model used to predict disaster losses and death tolls.
- **`random_forest_pred.py`**: (Optional) An experimental script to test Random Forest performance.
- **`constructe_2025_input.py`**: Bridge script that converts the Transformer output into the input format required by the XGBoost model.
- **`predict_2025_loss.py`**: The final script that generates the comprehensive prediction (Disaster Occurrence + Social Loss) for 2025.

---

## 🚀 Getting Started

### 1. Environment Setup

First, create and activate the virtual environment using the provided YAML file:

```bash
conda env create -f environment.yml
conda activate IMHCNH
```
````

### 2\. Data Preparation

Ensure your database files(processed NOAA and NRI) are placed inside the `dataset/` folder.

---

## 🛠 Usage Pipeline

Follow these steps strictly to generate the final predictions.

### Step 1: Feature Engineering

Process the raw data into a format suitable for model training.

```bash
python feature_engineering.py
```

_Output:_ Saves processed files to `processed_data/`.

### Step 2: Frequency Prediction (Time Series)

Run the Transformer model to predict how many disasters are likely to occur in 2025.

```bash
python transformer_prediction.py
```

### Step 3: Loss Model Training

Train the XGBoost model to learn the relationship between disaster features, economic loss, and casualties.

```bash
python XGBoost.py
```

### (Optional) Random Forest Experiment

If you wish to compare results or experiment with a different algorithm:

```bash
python random_forest_pred.py
```

### Step 4: Input Construction

Combine the frequency predictions (from the Transformer) to create a standardized input vector for the loss model.

```bash
python constructe_2025_input.py
```

### Step 5: Final Prediction

Generate the comprehensive report for 2025, combining disaster frequency with estimated social and economic losses.

```bash
python predict_2025_loss.py
```

### Optional: Shap Analysis.

```bash
python XGBoost_shap.py
```

### Optional: Risk Classficiation.

```bash
python risk.py
```

### Optional: Confidence Score.

```bash
python XGBoost+confidence.py
```
