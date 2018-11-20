import os
import logging
import datetime
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re

w_tokenizer = RegexpTokenizer('\w+')


def clean(text):
    content = text.lower()
    year_reg = ('(19|20)\d{2}(/\d{2})?', 'yyyy')
    dollar_reg = ('[^ ]?\$\d+([^ ]*\d+)?', 'chachinggg')
    plainnum_reg = (' \d+([^ ]*\d)?( |\.|,)', ' plainnum ')
    website_reg = ('http[^ ]*( |\n)', 'wwwebsite ')
    regex_subs = [year_reg, dollar_reg, plainnum_reg, website_reg]
    for sub in regex_subs:
        content = re.sub(sub[0], sub[1], content)
    return content


def get_all_words():
    parent = '../data/Reuters-50/'
    authors = os.listdir(parent)
    if os.path.exists('../train-data/all_text.txt'):
        print('All words were already extracted.... Overwriting')
    all_text = ''
    for author in authors:
        for file in os.listdir(os.path.join(parent, author)):
            with open(os.path.join(parent, author, file), 'r') as f:
                all_text += clean(f.read())

    with open('../train-data/all_text.txt', 'w') as f:
        f.write(all_text)
    return all_text


def configure_logger(modelname, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    logger = logging.getLogger(modelname)
    logger.setLevel(level)

    now = datetime.datetime.now()
    fname = modelname +'_' + str(now.day) + '-' + str(now.month) + '_' + str(now.hour) + '-' + str(now.minute) + '.log'
    fh = logging.FileHandler(os.path.join('../logs', fname))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
