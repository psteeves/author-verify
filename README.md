# author-verify

This project helps solve the problem of authorship verification. After being trained on a corpus of texts from a large set of authors, the model used can discriminate between pairs of texts that were written by the same author and pairs of texts that were written by different ones. It achieved over 97% accuracy on the hold-out set.

### Model

![alt text](https://cdn-images-1.medium.com/max/1200/1*XzVUiq-3lYFtZEW3XfmKqg.jpeg =100x100 "Siamese Network Architecture")

The networks used were Bi-LSTMs and three dense layers coupled using contrastive loss. Read more about contrastive loss [in this paper](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) by Hadsell, Chopra, and LeCun

### Data
The dataset was created from the [Reuters 50-50 dataset](https://archive.ics.uci.edu/ml/datasets/Reuter_50_50), which contains 100 texts from 50 different authors all about one subtopic from the larger Reuters RCV1 dataset. For Element AI employees with access to the datacenter, the dataset can be found on the /mnt drive in my home directory in the author-verify/data folder.

### Requirements
Python packages:
- tensorflow v1.0 >
- nltk

Hardware:
Model was run on 2GPUs with 12GB
