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


def get_words_author(author):
    parent = '../data/Reuters-50/'
    text = ''
    for file in os.listdir(os.path.join(parent, author)):
        with open(os.path.join(parent, author, file), 'r') as f:
            text += clean(f.read())
    return text


def validate_text_lengths(min_words = 200):
    parent_dir = '../data/Reuters-50'
    authors = os.listdir(parent_dir)
    long_texts = {}
    for author in authors:
        long_texts[author] = []
        for text in os.listdir(os.path.join(parent_dir, author)):
            with open(os.path.join(parent_dir, author, text)) as f:
                content = f.read()
                cleaned_content = clean(content)
                words = w_tokenizer.tokenize(cleaned_content)
                if len(words) > min_words:
                    start = np.random.choice(len(words) - min_words)
                    sample = words[start : start + min_words]
                    idx = list(map(lambda x: dic.get(x, 0), sample))
                    long_texts[author].append((idx, text))
    pickle.dump(long_texts, open('../models/long_texts','wb'))
    return long_texts


def author_embeddings():
    parent = '../data/Reuters-50/'
    authors = os.listdir(parent)
    embeddings = pickle.load(open('../models/embeddings', 'rb'))
    author_embeds = {}
    words = {}
    for author in authors:
        text = get_words_author(author)
        words = w_tokenizer.tokenize(text)
        author_embeds[author] = np.stack([embeddings[dic.get(w, 0)] for w in words]).mean(axis = 0)
    return author_embeds


def configure_logger(modelname, level=20):
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    logger = logging.getLogger(modelname)
    logger.setLevel(level)

    now = datetime.datetime.now()
    fname = modelname +'_' + str(now.day) + '-' + str(now.month) + '.log'
    fh = logging.FileHandler(os.path.join('../logs', fname))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
