import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer

w_tokenizer = RegexpTokenizer('\w+')


def easy_clean(text):
    tokenized = w_tokenizer.tokenize(text)
    return ' '.join(tokenized).lower()


def get_all_words():
    parent = 'data/Reuters-50/'
    authors = os.listdir(parent)
    if 'all_text.txt' in os.path.exists('train-data/all_text.txt':
        print('All words were already extracted.... Overwriting')
    all_text = ''
    for author in authors:
        for file in os.listdir(os.path.join(parent, author)):
            with open(os.path.join(parent, author, file), 'r') as f:
                all_text += f.read()

    clean_text = easy_clean(all_text)
    with open(os.path.join('train-data/all_text.txt', 'all_text.txt'), 'w') as f:
        f.write(clean_text)
    return clean_text
