import os
import sys
import dill
from exception import CustomException
from logger import logging

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
