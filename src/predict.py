import pandas as pd
import shap
import warnings
import os
import numpy as np
import utils 

warnings.filterwarnings('ignore')
def load_models():

    print("Loading all models and objects...")
    models = {}

    models['classification'] = utils.load_object("best_classification_model.joblib")

    models['label_encoder'] = utils.load_object("label_encoder.joblib")

    reg_targets = [
        'total_population_affected', 'total_fatalities', 
        'total_injuries', 'total_socio_economic_loss'
    ]
    for target in reg_targets:
        model_file = f"best_regression_model_{target}.joblib"
        models[target] = utils.load_object(model_file)
        
    try:

        models['shap_explainer'] = shap.TreeExplainer(models['classification'])
    except Exception as e:
        print(f"Warning: Could not create SHAP explainer. Is the model tree-based? Error: {e}")
        models['shap_explainer'] = None

    if any(v is None for v in models.values()):
        print("Error: One or more model files are missing. Please run train.py first.")
        return None
        
    print("All models loaded successfully.")
    return models

def predict_full_package(models, features_df):
    if not models:
        return [{"error": "Models are not loaded."}]

    try:
        clf_model = models['classification']
        le = models['label_encoder']
        
        class_pred_numeric = clf_model.predict(features_df)
        
        class_pred_string = le.inverse_transform(class_pred_numeric)
        
        class_pred_proba = clf_model.predict_proba(features_df)
        
        reg_predictions = {}
        for target, model in models.items():
            if target.startswith('total_'):
                reg_predictions[target] = model.predict(features_df)
        shap_values = None
        if models['shap_explainer']:
            shap_values = models['shap_explainer'](features_df)
        results = []
        for i in range(len(features_df)):
            pred_index_numeric = class_pred_numeric[i]
            
            confidence = class_pred_proba[i][pred_index_numeric]
            
            regression_output = {
                target: reg_predictions[target][i] for target in reg_predictions
            }
            explain_output = {"top_3_factors": [], "full_details": {}}
            if shap_values is not None:
                shap_row = shap_values[i]
                mean_abs_shap_values = np.abs(shap_row.values).mean(axis=1)
                shap_importances = pd.Series(mean_abs_shap_values, index=features_df.columns)
                top_3_factors = shap_importances.abs().nlargest(3).index.tolist()
                explain_output["top_3_factors"] = top_3_factors
                explain_output["full_details"] = shap_importances.to_dict()

            
            result_package = {
                "input_index": i,
                "classification": {
                    "rank": class_pred_string[i],
                    "confidence": f"{confidence:.2%}"
                },
                "regression_predictions": regression_output,
                "explainability": explain_output
            }
            results.append(result_package)
            
        return results

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return [{"error": str(e)}]