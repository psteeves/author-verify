import logging
import time

def configure_logger():
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    logger = logging.getLogger('test_application')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler('logs/test.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger

spec_logger = configure_logger()
print('Hello')

spec_logger.info('hello printed')
time.sleep(5)
spec_logger.info('5 sec later')
