import pandas as pd
import joblib
import numpy as np

def predict_new_cases(input_data):
    """
    Function to make predictions on new data
    input_data: Dictionary or DataFrame containing the same features as training data
    """
    # Load the saved models and scaler
    collective_model = joblib.load('xgb_collective_model.pkl')
    groups_model = joblib.load('xgb_groups_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Create DataFrame from input
    new_data = pd.DataFrame([input_data])
    
    # Standardize the numerical features (same as training)
    numerical_cols = ['Fold change of RAMP (logarithmic scale)', 
                     'Fold change of FENDRlogarithmic scale)',
                     'triglycerides', 'CKMB', 'Troponin', 'BMI', 'age']
    
    new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])
    
    # First prediction (control vs ACS)
    X_new = new_data.drop(columns=['groups', 'groups collective'], errors='ignore')
    collective_pred = collective_model.predict(X_new)
    
    # Prepare for second prediction
    X_new_second = X_new.copy()
    X_new_second['predicted_groups_collective'] = collective_pred
    
    # Second prediction (specific condition)
    groups_pred = groups_model.predict(X_new_second)
    
    # Map numeric predictions back to labels
    group_mapping = {
        0: 'healthy control',
        1: 'non cardiac chest pain',
        2: 'unstable angina',
        3: 'NSTEMI',
        4: 'STEMI'
    }
    
    collective_mapping = {
        0: 'control',
        1: 'acute coronary syndrome'
    }
    
    return {
        'primary_classification': collective_mapping[collective_pred[0]],
        'specific_diagnosis': group_mapping[groups_pred[0]]
    }


# Example input data
new_patient = {
    'Fold change of RAMP (logarithmic scale)': 8.25,
    'Fold change of FENDRlogarithmic scale)': 23.020,
    'triglycerides': 160,
    'CKMB': 26,
    'Troponin': 12,
    'BMI': 28,
    'age': 52
}

# Make prediction
result = predict_new_cases(new_patient)
print("\nPrediction Results:")
print(f"1. Primary Classification: {result['primary_classification']}")
print(f"2. Specific Diagnosis: {result['specific_diagnosis']}")
