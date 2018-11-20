import os
from copy import copy
from utils import configure_logger, clean
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import RegexpTokenizer
w_tokenizer = RegexpTokenizer('\w+')


sequence_length = 150
lstm_num_units = [512, 128]

dic = pickle.load(open('../models/dictionary', 'rb'))
stored_embeddings = pickle.load(open('../models/embeddings', 'rb'))


def create_data(chosen_author, min_words = sequence_length, verbose = False):
    """
    Input:
        min_words: min words in article for it to be in data
        num_refs: number of texts to use as references to compare to candidate text
    Output: CSV file containing filenames of texts to serve as features and labels corresponding to whether a candidate is from the same author
    """

    parent_dir = '../data/Reuters-50'
    authors = os.listdir(parent_dir)
    count = len(authors)
    c = 0

    # Discard texts shorter than min length. Do first seperately so we don't have to check length again when choosing candidates from other authors
    data = pd.DataFrame({}, columns = ['author', 'file', 'pos', 'words', 'target'])

    for author in authors:
        texts = os.listdir(os.path.join(parent_dir, author))
        if author != chosen_author:
            texts = np.random.choice(texts, 2, replace=False)
        for text in texts:
            with open(os.path.join(parent_dir,author,text)) as f:
                content = f.read()
                cleaned_content = clean(content)
                words = w_tokenizer.tokenize(cleaned_content)
                wc = len(words)
                if wc > min_words:
                    num_samples = wc // min_words
                    for pos in range(num_samples):
                        idx = list(map(lambda x: dic.get(x, 0), words[pos*min_words : (pos+1)*min_words]))
                        if author == chosen_author:
                            target = 1
                        else:
                            target = 0
                        row = [chosen_author, os.path.join(author, text), pos, idx, target]
                        data = data.append(dict(zip(data.columns, row)), ignore_index=True)
    return data


def generate_batch(data, batch_num, size):
    subset = data.iloc[batch_num*size : batch_num*size + size,:]
    candidates = np.stack(subset.loc[:,'words'].values)
    targets = subset.iloc[:,-1].values.reshape(size, 1)

    return candidates, targets


def get_accuracy(outs, labels):
    sigmoids = tf.sigmoid(outs)
    preds = tf.round(sigmoids)
    score = tf.equal(preds, labels)
    accuracy = tf.reduce_mean(tf.cast(score, tf.float32))
    return accuracy


def train(data, epochs = 10, batch_size = 64):
    logger = configure_logger(modelname = 'lstm_per_author_train')

    graph = tf.Graph()
    with graph.as_default():
        train_candidates = tf.placeholder(tf.int32, shape = (None, sequence_length))
        train_targets = tf.placeholder(tf.float32, shape = (None,1))
        embeddings = tf.constant(stored_embeddings, dtype = tf.float32)
        candidates_embed = tf.nn.embedding_lookup(embeddings, train_candidates)

        initializer = tf.initializers.truncated_normal()

        def lstm_cell(hidden_size):
            return tf.contrib.rnn.BasicLSTMCell(hidden_size, activation = tf.nn.relu)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(n) for n in lstm_num_units])
        LSTM_candidates_outs = tf.nn.dynamic_rnn(stacked_lstm, candidates_embed, dtype = tf.float32)

        last_states_candidates = LSTM_candidates_outs[1][len(lstm_num_units)-1].h

        layer1 = tf.layers.dense(last_states_candidates, 128, kernel_initializer = initializer, activation = tf.nn.relu)
        layer2 = tf.layers.dense(layer1, 32, kernel_initializer = initializer, activation = tf.nn.relu)
        outputs = tf.layers.dense(layer2, 1, kernel_initializer = initializer, activation = None)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_targets, logits=outputs))
        optimizer = tf.train.AdamOptimizer()
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        preds = tf.round(tf.sigmoid(outputs))
        accuracy = get_accuracy(outputs, train_targets)

        saver = tf.train.Saver()


    num_batches = len(data) // batch_size
    with tf.Session(graph = graph) as sess:
        tf.global_variables_initializer().run()
        cum_loss = 0
        for epoch in range(epochs):
            data = data.sample(frac=1)
            for batch in range(num_batches):
                batch_refs, batch_candidates, batch_targets = generate_batch(data, batch, batch_size)
                feed_dict = {t_ref: b_ref for t_ref, b_ref in zip(train_refs, batch_refs)}
                feed_dict.update({train_candidates: batch_candidates, train_targets: batch_targets})
                _, l = sess.run([train_op, loss], feed_dict=feed_dict)
                cum_loss += l
                if (batch + 1) % 400 == 0:
                    msg = 'Batch {} of {}. Avg loss past 400 batches: {:0.3f}'.format(batch + 1, num_batches, cum_loss/400)
                    print(msg)
                    logger.info(msg)
                    cum_loss = 0
            all_refs, all_candidates, all_targets = generate_batch(data, np.random.choice(5), int(len(data)/5))
            acc_feed_dict = {t_ref: b_ref for t_ref, b_ref in zip(train_refs, all_refs)}
            acc_feed_dict.update({train_candidates: all_candidates, train_targets: all_targets})
            msg = 'Done epoch {}. Accuracy on random fifth of training set: {:.1%}'.format(epoch, accuracy.eval(feed_dict=acc_feed_dict))
            print(msg)
            logger.info(msg)
        saver.save(sess, '../models/lstm/model')
        logger.info('Training finished. Saved model')


if __name__ == "__main__":
    #if os.path.exists('../train-data/data.csv'):
    #    data = pd.read_csv('../train-data/data.csv').iloc[:10000,:]
    data = create_data()

    train(data, epochs = 5)
