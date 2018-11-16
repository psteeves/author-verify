FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update -yq && apt-get install -yq vim && yes | pip install nltk

RUN mkdir -p /mnt/home/author-verify
WORKDIR /mnt/home/author-verify/scripts

#CMD bash
CMD ["python", "word2vec.py"]

