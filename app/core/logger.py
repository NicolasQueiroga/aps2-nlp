import os
import logging
from pythonjsonlogger import jsonlogger

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["level"] = record.levelname
        log_record["name"] = record.name
        log_record["message"] = record.getMessage()


logger = logging.getLogger("app_logger")
logger.setLevel(LOG_LEVEL)

console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)

json_formatter = CustomJsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
console_handler.setFormatter(json_formatter)

logger.addHandler(console_handler)

scheduler_logger = logging.getLogger("apscheduler")
scheduler_logger.setLevel(logging.INFO)
scheduler_logger.addHandler(console_handler)
