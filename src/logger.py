import logging 
import os
from datetime import datetime 
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%M%S')}.log"
log_path= os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(filename=log_path, level=logging.INFO)
logging.info("Training started")
