import logging
import torch
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir='logs', log_file='training.log'):
        self.log_dir = log_dir
        self.log_file = log_file
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)

        # Create log directory if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create file handler
        log_path = os.path.join(self.log_dir, self.log_file)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def critical(self, message):
        self.logger.critical(message)

def log_cuda_memory(logger):
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        max_memory_allocated = torch.cuda.max_memory_allocated()
        max_memory_reserved = torch.cuda.max_memory_reserved()

        logger.info(f"Allocated Memory: {allocated_memory / (1024 ** 2):.2f} MB")
        logger.info(f"Reserved Memory: {reserved_memory / (1024 ** 2):.2f} MB")
        logger.info(f"Max Allocated Memory: {max_memory_allocated / (1024 ** 2):.2f} MB")
        logger.info(f"Max Reserved Memory: {max_memory_reserved / (1024 ** 2):.2f} MB")
    else:
        logger.warning("CUDA is not available.")