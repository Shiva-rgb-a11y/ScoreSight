import os
import sys
import dill
from exception import CustomException
from logger import logging
from sklearn.metrics import r2_score

class ObjectSaver:
    """
    Utility class to save Python objects using dill.
    """

    @staticmethod
    def save(file_path, obj):
        """
        Save any Python object to disk.

        Args:
            file_path (str): Destination file path (.pkl).
            obj (Any): Object to be serialized.

        Raises:
            CustomException: If saving fails.
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)

            logging.info(f"Object saved successfully at: {file_path}")

        except Exception as e:
            logging.error("Error occurred while saving object", exc_info=True)
            raise CustomException(e, sys)
        

    def evaluate_model(x_train, y_train, x_test, y_test, models):
        try:
            report = {}
            for name, model in models.items():
                model.fit(x_train, y_train)  # Train model
                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                report[name] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)
