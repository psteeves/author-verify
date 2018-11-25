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


def get_accuracy(outs, labels):
    sigmoids = tf.sigmoid(outs)
    preds = tf.round(sigmoids)
    score = tf.equal(preds, labels)
    accuracy = tf.reduce_mean(tf.cast(score, tf.float32))
    return accuracy


def forward_pass(ref_embeds, cand_embeds):
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        initializer = tf.initializers.truncated_normal()
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_num_units, activation = tf.nn.tanh)
        LSTM_outs_ref = tf.nn.dynamic_rnn(lstm_cell, ref_embeds, dtype = tf.float32)
        LSTM_outs_cand = tf.nn.dynamic_rnn(lstm_cell, cand_embeds, dtype = tf.float32)
        last_states_ref = LSTM_outs_ref[1].h
        last_states_cand = LSTM_outs_cand[1].h
        all_states = tf.concat([last_states_ref, last_states_cand], axis = 1)
        mask = tf.layers.dense(last_states_ref, 2*sequence_length, kernel_initializer = initializer, activation = tf.nn.sigmoid)
        post_mask = mask * all_states
        drop0 = tf.nn.dropout(post_mask, 0.9)
        layer1 = tf.layers.dense(drop0, 128, kernel_initializer = initializer, activation = tf.nn.relu)
        drop1 = tf.nn.dropout(layer1, 0.8)
        layer2 = tf.layers.dense(drop1, 32, kernel_initializer = initializer, activation = tf.nn.relu)
        drop2 = tf.nn.dropout(layer2, 0.8)
        outputs = tf.layers.dense(drop2, 1, kernel_initializer = initializer, activation = None)
    return outputs


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

        outputs = forward_pass(train_refs_embed, train_candidates_embed)
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


