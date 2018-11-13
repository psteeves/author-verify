import os
import shutil

authors = os.listdir('data/C50test')    # List of all authors in dataset

# Combine articles into one folder to be split into train and test only later
for author in authors:
    origpath = 'data/C50test/'+author
    destpath = 'data/C50train/'+author
    files = os.listdir(origpath)
    for file in files:
        old = os.path.join(origpath, file)
        new = os.path.join(destpath, file)
        os.rename(old, new)

shutil.rmtree('data/C50test')
os.rename('data/C50train', 'data/Reuters-50')
