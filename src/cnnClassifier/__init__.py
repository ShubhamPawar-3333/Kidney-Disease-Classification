import os
import sys
import logging

class LoggerInitializer:
    LOG_FORMAT = "[%(asctime)s] %(module)s, line:%(lineno)s - %(levelname)s : %(message)s"
    LOG_DIR = "logs"
    LOG_FILEPATH = os.path.join(LOG_DIR, "project_logs.log")

    def __init__(self, logger_name):
        self.logger_name = logger_name

    def configure_logger(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format=self.LOG_FORMAT,
            handlers=[
                logging.FileHandler(self.LOG_FILEPATH),
                logging.StreamHandler(sys.stdout)
            ]
        )

        return logging.getLogger(self.logger_name)

logger_initializer = LoggerInitializer("cnnClassifierLogger")
logger = logger_initializer.configure_logger()
