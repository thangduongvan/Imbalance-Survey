from __future__ import absolute_import
import logging
from os.path import join, dirname, abspath
import os


def fast_log(msg, filename='res.log'):
    logging.basicConfig(filename=filename, level=logging.INFO)
    logging.info(msg)


def setup_logger(logger_name, log_file, level=logging.INFO):
    """Setup log file

    Args:
        logger_name ('str'): name of logger
        log_file (str): directory to log file
        level (logging type, optional): 'INFO', 'WARNING', etc . Defaults to logging.INFO.
    
    Usages:
        >> setup_logger('log1', r'C:\temp\log1.log')
        >> setup_logger('log2', r'C:\temp\log2.log')
        >> log1 = logging.getLogger('log1')
        >> log2 = logging.getLogger('log2')

        >> log1.info('Info for log 1!')
        >> log2.info('Info for log 2!')
        >> log1.error('Oh, no! Something went wrong!')
    """
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def clean_console():
    """Clear log in terminal when too many lines"""
    command = 'clear' # default is clear
    if os.name in ('nt', 'dos'): # if machine is running on windows, cls is used
        command = 'cls'
    os.system(command)