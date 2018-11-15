FROM tensorflow/tensorflow:latest-py3

RUN apt-get update -yq && apt-get install -yq vim && yes | pip install nltk

RUN mkdir -p /mnt/home/psteeves/author-verify
WORKDIR /mnt/home/psteeves/author-verify/scripts

#CMD bash
CMD ["python", "lstm.py"]
