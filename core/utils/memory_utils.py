import os
import psutil
from asyncio.log import logger


def get_memory_usage():
    # get current process
    process = psutil.Process(os.getpid())

    # get memory usage
    mem_info = process.memory_info()

    # return memory ussage
    return f"RSS={mem_info.rss / (1024 ** 2):.2f}, VMS={mem_info.vms / (1024 ** 2):.2f} MB"

def log_memory_usage(label=""):
        # log the memory usage
        logger.info(f"{label} - Memory Usage: {get_memory_usage()}")
