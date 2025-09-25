# Import standard libraries
import warnings
import os

# Import third-party libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Silence warnings in the UI
warnings.filterwarnings("ignore")

# Define training feature contracts
NUMERIC_FEATURES = ['age', 'bmi', 'children']
CATEGORICAL_FEATURES = ['sex', 'smoker', 'region']
ALL_FEATURES = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

# Define a loader for the fitted RandomForest pipeline
def load_pipeline():
    # Define fixed model path
    model_path = os.path.join("models", "RandomForest.joblib")
    # Load the artifact
    pipeline = joblib.load(model_path)
    # Return the pipeline
    return pipeline

# Define a helper to extract components from the pipeline
def extract_components(pipeline):
    # Get preprocess step
    preprocessor = pipeline.named_steps.get("preprocess", None)
    # Get regressor step
    regressor = pipeline.named_steps.get("regressor", None)
    # Get fitted OneHotEncoder from categorical branch
    ohe = None
    if preprocessor is not None and hasattr(preprocessor, "named_transformers_"):
        ohe = preprocessor.named_transformers_.get("cat", None)
    # Return components
    return preprocessor, ohe, regressor

# Define a manual transformer using the fitted OneHotEncoder
def manual_transform(input_df, ohe):
    # Select numeric values
    x_num = input_df[NUMERIC_FEATURES].to_numpy()
    # Select categorical values
    x_cat_df = input_df[CATEGORICAL_FEATURES]
    # Transform categoricals
    x_cat = ohe.transform(x_cat_df)
    # Convert sparse to dense if needed
    if hasattr(x_cat, "toarray"):
        x_cat = x_cat.toarray()
    # Concatenate numeric and categorical arrays
    x_all = np.hstack([x_num, x_cat])
    # Return combined features
    return x_all

# Define a quick fittedness check for regressors
def regressor_looks_fitted(regressor):
    # Check common fitted attributes
    if hasattr(regressor, "n_features_in_") or hasattr(regressor, "feature_importances_"):
        return True
    if hasattr(regressor, "coef_") or hasattr(regressor, "intercept_"):
        return True
    return False

# Define the Streamlit application
def main():
    # Set title
    st.title("💊 Medical Insurance Charges Predictor")
    # Set description
    st.write("""
Predict individual medical charges based on demographic and health information.
""")
    # Load the fitted RandomForest pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load models/RandomForest.joblib: {e}")
        return
    # Extract components from the pipeline
    try:
        preprocessor, ohe, regressor = extract_components(pipeline)
    except Exception as e:
        st.error(f"Failed to inspect pipeline: {e}")
        return
    # Validate components
    if regressor is None or not regressor_looks_fitted(regressor):
        st.error("Loaded RandomForest pipeline appears unfitted. Save a fitted artifact (e.g., best_rf) to models/RandomForest.joblib.")
        return
    if ohe is None or not hasattr(ohe, "categories_"):
        st.error("Missing fitted OneHotEncoder in the pipeline preprocess step. Ensure your saved pipeline includes the trained encoder.")
        return
    # Collect inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    # Build single-row dataframe
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }], columns=ALL_FEATURES)
    # Compute prediction live
    try:
        # Manually transform to bypass ColumnTransformer passthrough serialization issues
        x_infer = manual_transform(input_df, ohe)
        # Predict with fitted regressor
        pred = regressor.predict(x_infer)[0]
        # Show result
        st.success(f"💵 Estimated Medical Charges: ${pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Define script entrypoint
if __name__ == "__main__":
    main()
