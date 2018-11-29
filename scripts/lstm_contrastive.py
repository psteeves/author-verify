import os
from copy import copy
from utils import configure_logger, clean
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import ast
from nltk.tokenize import RegexpTokenizer
w_tokenizer = RegexpTokenizer('\w+')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sequence_length = 250
lstm_num_units = 1024

dic = pickle.load(open('../models/dictionary', 'rb'))
stored_embeddings = pickle.load(open('../models/embeddings', 'rb'))


def balance_classes(data):
    positives = data[data.target == 1]
    negatives = data[data.target == 0]
    max_len = min(len(positives), len(negatives))

    neg_subsample = negatives.sample(n = max_len)
    pos_subsample = positives.sample(n = max_len)
    balanced = pd.concat([neg_subsample, pos_subsample]).sample(frac=1).reset_index(drop=True)
    return balanced


def validate_text_lengths(min_words):
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
    return long_texts


def create_data(num_authors, split = [0.7,0.85], min_words = sequence_length, replace=True):
    """
    Input:
        min_words: min words in article for it to be in data
        num_refs: number of texts to use as references to compare to candidate text
    Output: CSV file containing filenames of texts to serve as features and labels corresponding to whether a candidate is from the same author
    """
    path = '../train-data/data_' + str(num_authors) + '.csv'
    if replace:
        c = 0

        data = pd.DataFrame({}, columns = ['author', 'ref_words', 'ref_file', 'other_author', 'cand_words', 'cand_file', 'target'])
        long_texts = validate_text_lengths(min_words)
        print('Done validating texts')
        authors = list(np.random.choice(list(long_texts.keys()), num_authors))
        other_authors = copy(authors)
        for author in authors:
            other_authors.remove(author)
            options = copy(long_texts[author])
            for text in long_texts[author]:
                options.remove(text)
                ref_words = text[0]
                ref_file = text[1]
                for option in options:
                    cand_words = option[0]
                    cand_file = option[1]
                    hit = dict(zip(data.columns, [author, ref_words, ref_file, author, cand_words, cand_file, 1]))
                    data = data.append(hit, ignore_index=True)

                for other_author in other_authors:
                    cands = [long_texts[other_author][i] for i in np.random.choice(len(long_texts[other_author]), 3, replace=False)]
                    for cand in cands:
                        cand_words = cand[0]
                        cand_file = cand[1]
                        miss = dict(zip(data.columns, [author, ref_words, ref_file, other_author, cand_words, cand_file, 0]))
                        data = data.append(miss, ignore_index=True)
            print('Done author ', c)
            c+=1

        data = data.sample(frac=1)
        data = balance_classes(data)
        data.to_csv(path, index=False)

    else:
        if os.path.exists(path):
            data = pd.read_csv(path)
            data.loc[:,['ref_words', 'cand_words']] = data.loc[:,['ref_words', 'cand_words']].applymap(ast.literal_eval)
        else:
            raise ValueError('Data file {} not found'.format(path))

    valid_split, test_split = int(split[0]*len(data)), int(split[1]*len(data))
    train_data = data.iloc[:valid_split,:]
    valid_data = data.iloc[valid_split:test_split,:]
    test_data = data.iloc[test_split:,:]
    return train_data, valid_data, test_data


def generate_batch(data, batch_num, size):
    subset = data.iloc[batch_num*size : batch_num*size + size,:]
    refs = np.stack(subset.loc[:,'ref_words'].values)
    candidates = np.stack(subset.loc[:,'cand_words'].values)
    targets = subset.iloc[:,-1].values.reshape(size, 1)
    return refs, candidates, targets


def get_accuracy(refs_out, candidates_out, threshold, labels):
    dist = tf.sqrt(tf.reduce_sum(tf.square(refs_out - candidates_out), 1, keepdims = True))
    preds = tf.cast(tf.less(dist, threshold), tf.float32)
    score = tf.equal(preds, labels)
    accuracy = tf.reduce_mean(tf.cast(score, tf.float32))
    return accuracy


def forward_pass(embeds):
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        initializer = tf.initializers.truncated_normal()
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_num_units, activation = tf.nn.tanh) for _ in range(2)])
        #lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_num_units, activation = tf.nn.tanh)
        LSTM_outs = tf.nn.dynamic_rnn(stacked_lstm, embeds, dtype = tf.float32)
        last_states = LSTM_outs[-1][1].h
        drop0 = tf.nn.dropout(last_states, 1)
        layer1 = tf.layers.dense(drop0, 512, kernel_initializer = initializer, activation = tf.nn.relu)
        drop1 = tf.nn.dropout(layer1, 1)
        outputs = tf.layers.dense(drop1, 128, kernel_initializer = initializer, activation = tf.nn.relu)
        #outputs_norm = tf.nn.l2_normalize(outputs, axis = 1)
    return outputs


def train(train_data, valid_data, test_data, epochs = 10, batch_size = 128, return_results = False):
    logger = configure_logger(modelname = 'lstm_contrastive', level=10)

    graph = tf.Graph()
    with graph.as_default():
        embeddings = tf.constant(stored_embeddings, dtype = tf.float32)

        train_candidates = tf.placeholder(tf.int32, shape = (None, sequence_length))
        train_candidates_embed = tf.nn.embedding_lookup(embeddings, train_candidates)
        train_refs = tf.placeholder(tf.int32, shape = (None, sequence_length))
        train_refs_embed = tf.nn.embedding_lookup(embeddings, train_refs)
        train_targets = tf.placeholder(tf.float32, shape = (None,1))

        valid_candidates = tf.constant(generate_batch(valid_data, 0, len(valid_data))[0], dtype=tf.int32)
        valid_candidates_embed = tf.nn.embedding_lookup(embeddings, valid_candidates)
        valid_refs = tf.constant(generate_batch(valid_data, 0 , len(valid_data))[1], dtype = tf.int32)
        valid_refs_embed = tf.nn.embedding_lookup(embeddings, valid_refs)
        valid_targets = tf.constant(generate_batch(valid_data, 0, len(valid_data))[2], dtype=tf.float32)

        test_candidates = tf.constant(generate_batch(test_data, 0, len(test_data))[0], dtype=tf.int32)
        test_candidates_embed = tf.nn.embedding_lookup(embeddings, test_candidates)
        test_refs = tf.constant(generate_batch(test_data, 0, len(test_data))[1], dtype = tf.int32)
        test_refs_embed = tf.nn.embedding_lookup(embeddings, test_refs)
        test_targets = tf.constant(generate_batch(test_data, 0, len(test_data))[2], dtype=tf.float32)

        train_subset = train_data.sample(frac = 0.4)
        all_train_candidates = tf.constant(generate_batch(train_subset, 0, len(train_subset))[0], dtype=tf.int32)
        all_train_candidates_embed = tf.nn.embedding_lookup(embeddings, all_train_candidates)
        all_train_refs = tf.constant(generate_batch(train_subset, 0, len(train_subset))[1], dtype=tf.int32)
        all_train_refs_embed = tf.nn.embedding_lookup(embeddings, all_train_refs)
        all_train_targets = tf.constant(generate_batch(train_subset, 0, len(train_subset))[2], dtype=tf.float32)

        refs_outputs = forward_pass(train_refs_embed)
        candidates_outputs = forward_pass(train_candidates_embed)

        margin = tf.constant(5.)
        threshold = 1/2 * margin

        d = tf.sqrt(tf.reduce_sum(tf.square(refs_outputs - candidates_outputs), 1, keepdims = True))
        loss = tf.reduce_mean(train_targets * d + (1. - train_targets) * tf.maximum(0., margin - d))
        #loss = tf.contrib.losses.metric_learning.contrastive_loss(labels=train_targets, embeddings_anchor=refs_outputs, embeddings_positive=candidates_outputs, margin = margin)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        train_accuracy = get_accuracy(forward_pass(all_train_refs_embed), forward_pass(all_train_candidates_embed), threshold, all_train_targets)
        valid_accuracy = get_accuracy(forward_pass(valid_refs_embed), forward_pass(valid_candidates_embed), threshold, valid_targets)
        test_accuracy = get_accuracy(forward_pass(test_refs_embed), forward_pass(test_candidates_embed), threshold, test_targets)

        saver = tf.train.Saver()

    num_batches = len(train_data) // batch_size
    with tf.Session(graph = graph) as sess:
        tf.global_variables_initializer().run()
        best_acc = 0
        for epoch in range(epochs):
            train_data = train_data.sample(frac=1)
            for batch in range(num_batches):
                batch_candidates, batch_refs, batch_targets = generate_batch(train_data, batch, batch_size)
                feed_dict = {train_candidates: batch_candidates, train_refs: batch_refs, train_targets: batch_targets, learning_rate: 0.001 * 0.97**epoch}
                _, l = sess.run([train_op, loss], feed_dict=feed_dict)
                msg = 'Batch {} of {}. Loss: {:0.3f}'.format(batch + 1, num_batches, l)
                logger.info(msg)
                if np.isnan(l):
                    print('Broken')
                    break

            valid_acc = valid_accuracy.eval()
            msg = 'Done epoch {}. '.format(epoch)
            msg += 'Validation accuracy: {:.1%} '.format(valid_acc)
            msg += 'Training accuracy: {:.1%} '.format(train_accuracy.eval())
            logger.info(msg)
            print(msg)

            if valid_acc > best_acc:
                saver.save(sess, '../models/lstm-any-author/model')
                best_acc = valid_acc

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, "../models/lstm-any-author/model")
        test_acc = test_accuracy.eval()
        logger.info('Test set accuracy: {:.1%} '.format(test_acc))
        if return_results:
            return {'Test accuracy': test_acc}


def run_model(num_authors = 30):
    print('Loading data')
    split = [0.7, 0.85]
    train_data, valid_data, test_data = create_data(num_authors = num_authors, split = split, replace = False)
    print('Data loaded')
    output = train(train_data, valid_data, test_data, epochs = 20, return_results = True)
    return output


if __name__ == "__main__":
    output = run_model(5)
