import os
import shutil


def restructure_data():
    '''
    Combine pre-defined train and test sets into one. Data will be split later during training
    '''
    
    authors = os.listdir('../data/C50test')    # List of all authors in dataset

    for author in authors:
        origpath = '../data/C50test/'+author
        destpath = '../data/C50train/'+author
        files = os.listdir(origpath)
        for file in files:
            old = os.path.join(origpath, file)
            new = os.path.join(destpath, file)
            os.rename(old, new)

    shutil.rmtree('../data/C50test')
    os.rename('../data/C50train', '../data/Reuters-50')

if __name__ == "__main__":
    restructure_data()
