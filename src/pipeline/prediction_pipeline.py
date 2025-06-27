import sys
import os
import pandas as pd
from exception import CustomException
from utils import ObjectSaver


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            # Load model and preprocessor from artifacts
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = ObjectSaver.load_object(file_path=model_path)
            preprocessor = ObjectSaver.load_object(file_path=preprocessor_path)

            # Handle missing values by replacing with "Unknown"
            features = features.fillna("Unknown")

            # Optional: align column order if preprocessor has feature_names_in_
            if hasattr(preprocessor, "feature_names_in_"):
                features = features[preprocessor.feature_names_in_]

            # Transform input
            data_scaled = preprocessor.transform(features)

            # Predict
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(f"Error during prediction: {e}", sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        # Defensive handling for None or empty fields
        self.gender = gender or "Unknown"
        self.race_ethnicity = race_ethnicity or "Unknown"
        self.parental_level_of_education = parental_level_of_education or "Unknown"
        self.lunch = lunch or "Unknown"
        self.test_preparation_course = test_preparation_course or "Unknown"
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(f"Error creating DataFrame from input data: {e}", sys)
