import os
from copy import copy
from utils import configure_logger, clean
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import RegexpTokenizer
w_tokenizer = RegexpTokenizer('\w+')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lstm_num_units = 512

dic = pickle.load(open('../models/dictionary', 'rb'))
stored_embeddings = pickle.load(open('../models/embeddings', 'rb'))


def create_data(chosen_author, split = [0.7,0.85], min_words = sequence_length, verbose = False):
    """
    Input:
        min_words: min words in article for it to be in data
        num_refs: number of texts to use as references to compare to candidate text
    Output: CSV file containing filenames of texts to serve as features and labels corresponding to whether a candidate is from the same author
    """

    parent_dir = '../data/Reuters-50'
    authors = os.listdir(parent_dir)
    if chosen_author not in authors:
        raise ValueError('Author not found')

    data = pd.DataFrame({}, columns = ['author', 'file', 'words', 'avg_w_len', 'unique_w_ratio', 'pos', 'target'])

    for author in authors:
        texts = os.listdir(os.path.join(parent_dir, author))
        if author != chosen_author:
            texts = np.random.choice(texts, 4, replace=False)
        for text in texts:
            with open(os.path.join(parent_dir,author,text)) as f:
                content = f.read()
                cleaned_content = clean(content)
                words = w_tokenizer.tokenize(cleaned_content)
                wc = len(words)
                if wc > min_words:
                    num_samples = (wc-min_words) // (min_words - overlap) + 1
                    for pos in range(num_samples):
                        sample = words[pos * (min_words-overlap) : pos * (min_words-overlap) + min_words]
                        idx = list(map(lambda x: dic.get(x, 0), sample))

                        avg_len = np.mean([len(w) for w in words]) - 5
                        unique_ratio = len(set(words)) / len(words) - 0.5
                        if author == chosen_author:
                            target = 1
                        else:
                            target = 0
                        row = [chosen_author, os.path.join(author, text), idx, avg_len, unique_ratio, pos, target]
                        data = data.append(dict(zip(data.columns, row)), ignore_index=True)

    data = data.sample(frac=1)
    valid_split, test_split = int(split[0]*len(data)), int(split[1]*len(data))

    train_data = data.iloc[:valid_split,:]
    valid_data = data.iloc[valid_split:test_split,:]
    test_data = data.iloc[test_split:,:]
    return train_data, valid_data, test_data


def generate_batch(data, batch_num, size):
    subset = data.iloc[batch_num*size : batch_num*size + size,:]
    candidates = np.stack(subset.loc[:,'words'].values)
    hand_features = subset.loc[:,['avg_w_len', 'unique_w_ratio']].values
    targets = subset.iloc[:,-1].values.reshape(size, 1)
    return candidates, hand_features, targets


def get_accuracy(outs, labels):
    sigmoids = tf.sigmoid(outs)
    preds = tf.round(sigmoids)
    score = tf.equal(preds, labels)
    accuracy = tf.reduce_mean(tf.cast(score, tf.float32))
    return accuracy


def forward_pass(embeds, hand_features):
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        initializer = tf.initializers.truncated_normal()
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_num_units, activation = tf.nn.tanh)
        LSTM_candidates_outs = tf.nn.dynamic_rnn(lstm_cell, embeds, dtype = tf.float32)
        last_states_candidates = LSTM_candidates_outs[1].h
        drop0 = tf.nn.dropout(last_states_candidates, 0.8)
        layer1 = tf.layers.dense(drop0, 128, kernel_initializer = initializer, activation = tf.nn.relu)
        drop1 = tf.nn.dropout(layer1, 0.7)
        add_features = tf.concat([drop1, hand_features], axis = 1)
        layer2 = tf.layers.dense(add_features, 32, kernel_initializer = initializer, activation = tf.nn.relu)
        drop2 = tf.nn.dropout(layer2, 0.7)
        outputs = tf.layers.dense(drop2, 1, kernel_initializer = initializer, activation = None)
    return outputs


def train(author, train_data, valid_data, test_data, epochs = 10, batch_size = 64, return_results = False):
    logger = configure_logger(modelname = 'lstm_'+author, level=10)
    sequence_length = len(train_data.iloc[0,1])
    graph = tf.Graph()
    with graph.as_default():
        embeddings = tf.constant(stored_embeddings, dtype = tf.float32)

        train_candidates = tf.placeholder(tf.int32, shape = (None, sequence_length))
        train_candidates_embed = tf.nn.embedding_lookup(embeddings, train_candidates)
        train_hand_features = tf.placeholder(tf.float32, shape = (None, 2))
        train_targets = tf.placeholder(tf.float32, shape = (None,1))

        valid_candidates = tf.constant(generate_batch(valid_data, 0, len(valid_data))[0], dtype=tf.int32)
        valid_candidates_embed = tf.nn.embedding_lookup(embeddings, valid_candidates)
        valid_hand_features = tf.constant(generate_batch(valid_data, 0, len(valid_data))[1], dtype=tf.float32)
        valid_targets = tf.constant(generate_batch(valid_data, 0, len(valid_data))[2], dtype=tf.float32)

        test_candidates = tf.constant(generate_batch(test_data, 0, len(test_data))[0], dtype=tf.int32)
        test_candidates_embed = tf.nn.embedding_lookup(embeddings, test_candidates)
        test_hand_features = tf.constant(generate_batch(test_data, 0, len(test_data))[1], dtype=tf.float32)
        test_targets = tf.constant(generate_batch(test_data, 0, len(test_data))[2], dtype=tf.float32)

        all_train_candidates = tf.constant(generate_batch(train_data, 0, len(train_data))[0], dtype=tf.int32)
        all_train_candidates_embed = tf.nn.embedding_lookup(embeddings, all_train_candidates)
        all_train_hand_features = tf.constant(generate_batch(train_data, 0, len(train_data))[1], dtype=tf.float32)
        all_train_targets = tf.constant(generate_batch(train_data, 0, len(train_data))[2], dtype=tf.float32)

        outputs = forward_pass(train_candidates_embed, train_hand_features)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_targets, logits=outputs))
        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        preds = tf.round(tf.sigmoid(outputs))

        train_accuracy = get_accuracy(forward_pass(all_train_candidates_embed, all_train_hand_features), all_train_targets)
        valid_accuracy = get_accuracy(forward_pass(valid_candidates_embed, valid_hand_features), valid_targets)
        test_accuracy = get_accuracy(forward_pass(test_candidates_embed, test_hand_features), test_targets)

        saver = tf.train.Saver()

    num_batches = len(train_data) // batch_size
    with tf.Session(graph = graph) as sess:
        tf.global_variables_initializer().run()
        best_acc = 0
        for epoch in range(epochs):
            train_data = train_data.sample(frac=1)
            for batch in range(num_batches):
                batch_candidates, batch_hand_features, batch_targets = generate_batch(train_data, batch, batch_size)
                feed_dict = {train_candidates: batch_candidates, train_hand_features: batch_hand_features, train_targets: batch_targets, learning_rate: 0.001 * 0.97**epoch}
                _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
                msg = 'Batch {} of {}. Loss: {:0.3f}'.format(batch + 1, num_batches, l)
                logger.info(msg)

            valid_acc = valid_accuracy.eval()
            msg = 'Done epoch {}. '.format(epoch)
            msg += 'Validation accuracy: {:.1%} '.format(valid_acc)
            msg += 'Training accuracy: {:.1%} '.format(train_accuracy.eval())
            logger.info(msg)
            if valid_acc > best_acc:
                saver.save(sess, '../models/lstm-per-author/model')
                best_acc = valid_acc

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, "../models/lstm-per-author/model")
        test_acc = test_accuracy.eval()
        logger.info('Test set accuracy: {:.1%} '.format(test_acc))
        if return_results:
            return {'Data quantity': len(train_data), 'Class balance': train_data.target.mean(), 'Test accuracy': test_acc}

def run_model(author, sequence_length = 100):
    train_data, valid_data, test_data = create_data(author, split = [0.7, 0.85], sequence_length = sequence_length)
    output = train(author, train_data, valid_data, test_data, epochs = 20, return_results = True)
    return output

if __name__ == "__main__":
    run_model('ScottHillis')
