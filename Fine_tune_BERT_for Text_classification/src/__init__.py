import os
import sys
import logging

logging_fmt = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_file = os.path.join(log_dir, "running_logs.log")

if not os.path.exists(log_dir): os.makedirs(log_dir)

logging.basicConfig(
    level = logging.INFO,
    format = logging_fmt,
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)