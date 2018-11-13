import os
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string
from collections import Counter

w_tokenizer = RegexpTokenizer('\w+')


def easy_clean(text):
    tokenized = w_tokenizer.tokenize(text)
    return ' '.join(tokenized).lower()


def get_all_words():
    parent = 'data/Reuters-50/'
    authors = os.listdir(parent)
    if 'all_text.txt' in authors:
        authors = list(set(authors) - {'all_text.txt'})
        print('All words were already extracted.... Overwriting')
    all_text = ''
    for author in authors:
        files = os.listdir(os.path.join(parent, author))
        if 'all_text.txt' in files:
            files = list(set(files) - {'all_text.txt'})
        for file in files:
            with open(os.path.join(parent, author, file), 'r') as f:
                all_text += f.read()

    clean_text = easy_clean(all_text)
    with open(os.path.join(parent, 'all_text.txt'), 'w') as f:
        f.write(clean_text)
    return clean_text
