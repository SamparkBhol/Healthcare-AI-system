import logging
import os

class LoggerConfig:
    def __init__(self, log_dir="logs", log_file="training.log", log_level=logging.INFO):
        self.log_dir = log_dir
        self.log_file = log_file
        self.log_level = log_level
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logging.getLogger("Healthcare_Agent_Logger")
        self.logger.setLevel(self.log_level)

        # Create a file handler
        log_path = os.path.join(self.log_dir, self.log_file)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(self.log_level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

if __name__ == "__main__":
    logger_config = LoggerConfig()
    logger = logger_config.get_logger()

    logger.info("Training started...")
    logger.warning("This is a warning message.")
    logger.error("An error occurred during agent interaction.")
