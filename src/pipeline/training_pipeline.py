import os
import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from utils import ObjectSaver
from exception import CustomException


def train_pipeline(data_path: str):
    try:
        # Load dataset
        df = pd.read_csv(data_path)

        # Split features and target
        X = df.drop("math_score", axis=1)
        y = df["math_score"]

        # Define column types
        categorical_cols = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course"
        ]
        numerical_cols = ["reading_score", "writing_score"]

        # Handle missing values (optional, but safe)
        X[categorical_cols] = X[categorical_cols].fillna("Unknown")
        X[numerical_cols] = X[numerical_cols].fillna(0)

        # Preprocessing pipelines
        cat_pipeline = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown='ignore'))
        ])

        num_pipeline = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer([
            ("cat", cat_pipeline, categorical_cols),
            ("num", num_pipeline, numerical_cols)
        ])

        # Final pipeline
        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model_pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = model_pipeline.predict(X_test)
        score = r2_score(y_test, y_pred)
        print(f"Model R2 Score: {score:.2f}")

        # Save preprocessor and model separately
        os.makedirs("artifacts", exist_ok=True)
        ObjectSaver.save_object("artifacts/preprocessor.pkl", model_pipeline.named_steps["preprocessor"])
        ObjectSaver.save_object("artifacts/model.pkl", model_pipeline.named_steps["model"])

        print("✅ Model and preprocessor saved successfully in 'artifacts/' folder.")

    except Exception as e:
        raise CustomException(f"Training pipeline failed: {e}", sys)


if __name__ == "__main__":
    # Use your actual data file path here
    train_pipeline(data_path="notebooks/data/stud.csv")
