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

sequence_length = 100
overlap = 0
lstm_num_units = 512

dic = pickle.load(open('../models/dictionary', 'rb'))
stored_embeddings = pickle.load(open('../models/embeddings', 'rb'))

def validate_text_lengths(min_words = sequence_length):
    long_texts = {}
    for author in authors:
        for text in os.listdir(os.path.join(parent_dir, author)):
            with open(os.path.join(parent_dir, author, text)) as f:
                content = f.read()
                cleaned_content = clean(content)
                words = w_tokenizer.tokenize(cleaned_content)
                if len(words) > min_words:
                    long_texts[author].append(words)
    
    return long_texts


def create_data(split = [0.7,0.85], min_words = sequence_length, dups = 2, verbose = False):
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

    data = pd.DataFrame({}, columns = ['author', 'ref', 'candidate', 'target'])
    long_texts = validate_text_lengths()
    for author in authors:
        texts = long_texts[author]
        for text in texts:
            for i in range(dups):
                candidate = text
                ref = np.random.choice(list(set(texts) - set(candidate)))
                hit = dict(zip(data.columns, [author, ref, candidate, 1]))
                data.append(hit, ignore_index=True)

        for other_author in list(set(authors) - set(author)):
            for i in range(2*dups):
                ref = np.random.choice(texts)
                candidate = np.random.choice(long_texts[other_author])
                miss = dict(zip(data.columns, [author, ref, candidate, 0]))
                data.append(miss, ignore_index=True)

    data = data.sample(frac=1)
    data.to_csv('../train-data/data.csv')
    valid_split, test_split = int(split[0]*len(data)), int(split[1]*len(data))

    train_data = data.iloc[:valid_split,:]
    valid_data = data.iloc[valid_split:test_split,:]
    test_data = data.iloc[test_split:,:]
    return train_data, valid_data, test_data


def generate_batch(data, batch_num, size):
    subset = data.iloc[batch_num*size : batch_num*size + size,:]
    refs = np.stack(subset.loc[:,'ref'].values)
    candidates = np.stack(subset.loc[:,'candidate'].values)
    targets = subset.iloc[:,-1].values.reshape(size, 1)
    return refs, candidates, targets


def get_accuracy(refs_outs, candidates_out, margin, labels):
    dist = tf.sqrt(tf.reduce_sum(tf.square(refs_out - candidates_out), 1))
    preds = tf.cast(tf.less(dist, margin), tf.float32)
    score = tf.equal(preds, labels)
    accuracy = tf.reduce_mean(tf.cast(score, tf.float32))
    return accuracy


def forward_pass(embeds):
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        initializer = tf.initializers.truncated_normal()
        stacked_lstm = tf.nn.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_num_units, activation = tf.nn.tanh) for _ in range(2)]
        LSTM_outs = tf.nn.dynamic_rnn(stacked_lstm, embeds, dtype = tf.float32)
        last_states = LSTM_outs[1].h
        drop0 = tf.nn.dropout(last_states, 0.9)
        layer1 = tf.layers.dense(drop0, 128, kernel_initializer = initializer, activation = tf.nn.relu)
        drop1 = tf.nn.dropout(layer1, 0.8)
        layer2 = tf.layers.dense(drop1, 32, kernel_initializer = initializer, activation = tf.nn.relu)
        drop2 = tf.nn.dropout(layer2, 0.8)
        outputs = tf.layers.dense(drop2, 1, kernel_initializer = initializer, activation = None)
        outputs_norm = tf.math.l2_normalize(outputs, axis = 1)
    return outputs_norm


def train(author, train_data, valid_data, test_data, epochs = 10, batch_size = 64, return_results = False):
    logger = configure_logger(modelname = 'lstm_'+author, level=10)

    graph = tf.Graph()
    with graph.as_default():
        embeddings = tf.constant(stored_embeddings, dtype = tf.float32)

        train_candidates = tf.placeholder(tf.int32, shape = (None, sequence_length))
        train_candidates_embed = tf.nn.embedding_lookup(embeddings, train_candidates)
        train_refs = tf.placeholder(tf.float32, shape = (None, sequence_length))
        train_refs_embed = tf.nn.embedding_lookup(embeddings, train_refs)
        train_targets = tf.placeholder(tf.float32, shape = (None,1))

        valid_candidates = tf.constant(generate_batch(valid_data, 0, len(valid_data))[0], dtype=tf.int32)
        valid_candidates_embed = tf.nn.embedding_lookup(embeddings, valid_candidates)
        valid_refs = tf.constant(generate_batch(valid_data, 0 , len(valid_data))[1], dtype = tf.int32)
        valid_refs_embed = tf.nn.embeddings_lookup(embeddings, valid_refs)
        valid_targets = tf.constant(generate_batch(valid_data, 0, len(valid_data))[2], dtype=tf.float32)

        test_candidates = tf.constant(generate_batch(test_data, 0, len(test_data))[0], dtype=tf.int32)
        test_candidates_embed = tf.nn.embedding_lookup(embeddings, test_candidates)
        test_refs = tf.constant(generate_batch(test_data, 0, len(test_data))[1], dtype = tf.int32)
        test_refs_embed = tf.nn.embedding_lookup(embeddings, test_refs)
        test_targets = tf.constant(generate_batch(test_data, 0, len(test_data))[2], dtype=tf.float32)

        all_train_candidates = tf.constant(generate_batch(train_data, 0, len(train_data))[0], dtype=tf.int32)
        all_train_candidates_embed = tf.nn.embedding_lookup(embeddings, all_train_candidates)
        all_train_refs = tf.constant(generate_batch(train_data, 0, len(train_data))[1], dtype=tf.float32)
        all_train_refs_embed = tf.nn.embeddings_lookup(embeddings, all_train_refs)
        all_train_targets = tf.constant(generate_batch(train_data, 0, len(train_data))[2], dtype=tf.float32)

        refs_outputs = forward_pass(train_refs_embed)
        candidates_outputs = forward_pass(train_candidates_embed)

        margin = tf.constant(1.0)
        loss = tf.contrib.losses.metric_learning.contrastive_loss(labels=train_targets, embeddings_anchor=refs_outputs, embeddings_positive=candidates_outputs, margin = margin)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        train_accuracy = get_accuracy(forward_pass(all_train_refs_embed), forward_pass(all_train_candidates_embed), margin, all_train_targets)
        valid_accuracy = get_accuracy(forward_pass(valid_refs_embed), forward_pass(valid_candidates_embed), margin, valid_targets)
        test_accuracy = get_accuracy(forward_pass(test_refs_embed), forward_pass(test_candidates_embed), margin, test_targets)

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
                saver.save(sess, '../models/lstm-any-author/model')
                best_acc = valid_acc

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, "../models/lstm-any-author/model")
        test_acc = test_accuracy.eval()
        logger.info('Test set accuracy: {:.1%} '.format(test_acc))
        if return_results:
            return {'Test accuracy': test_acc}

def run_model():
    split = [0.7, 0.85]
    if os.path.exists('../train-data/data.csv'):
        data = pd.read_csv('../train-data/data.csv')
        valid_split, test_split = int(split[0]*len(data)), int(split[1]*len(data))
        train_data, valid_data, test_data = data.iloc[:valid_split, :], data.iloc[valid_split:test_split, :], data.iloc[test_split:, :]
    else:
        train_data, valid_data, test_data = create_data(author, split = split)
    output = train(author, train_data, valid_data, test_data, epochs = 20, return_results = True)
    return output

if __name__ == "__main__":
    output = run_model()


