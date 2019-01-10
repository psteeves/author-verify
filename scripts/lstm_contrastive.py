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

sequence_length = 200    # Length of text samples to take
dic = pickle.load(open('../models/dictionary', 'rb'))    # Dictionary of words for embeddings
stored_embeddings = pickle.load(open('../models/embeddings', 'rb'))    # Pre-trained embeddings


def balance_classes(data):
    '''
    Balance training data by target classes
    '''
    positives = data[data.target == 1]
    negatives = data[data.target == 0]
    max_len = min(len(positives), len(negatives))

    neg_subsample = negatives.sample(n = max_len)
    pos_subsample = positives.sample(n = max_len)
    balanced = pd.concat([neg_subsample, pos_subsample]).sample(frac=1).reset_index(drop=True)
    return balanced


def create_data(num_authors, spec_authors = None, min_words = sequence_length, split = 0.99, replace=False):
    """
    Create training data
    Input:
        num_authors: number of authors to select texts from in training data
        spec_authors: specific authors to create set from
        min_words: don't consider articles shorter than sequence length to be extracted
        split: training/valid split
        replace: whether to replace training data or load pre-created data from CSV
    Output: Training and validation data of pairs of texts tagged as written by same author or not
    """

    path = '../train-data/data_' + str(num_authors) + '.csv'

    if replace:
        data = pd.DataFrame({}, columns = ['author', 'ref_words', 'ref_file', 'other_author', 'cand_words', 'cand_file', 'target'])    # Empty df to append to iteratively
        long_texts = pickle.load(open('../models/long_texts', 'rb'))    # Load pre-filtered texts by length
        if spec_authors:
            authors = spec_authors
        else:
            authors = list(np.random.choice(list(long_texts.keys()), num_authors, replace = False))

        other_authors = copy(authors)    # other authors to sample negative examples from
        for author in authors:
            other_authors.remove(author)    # Don't consider this author for negative examples
            options = copy(long_texts[author])
            for text in long_texts[author]:
                options.remove(text)
                ref_words = text[0]
                ref_file = text[1]

                # Positive examples from same author
                for option in options:
                    cand_words = option[0]
                    cand_file = option[1]
                    hit = dict(zip(data.columns, [author, ref_words, ref_file, author, cand_words, cand_file, 1]))
                    data = data.append(hit, ignore_index=True)

                # Negative examples from other authors
                for other_author in other_authors:
                    cands = [long_texts[other_author][i] for i in np.random.choice(len(long_texts[other_author]), 20, replace=False)]
                    for cand in cands:
                        cand_words = cand[0]
                        cand_file = cand[1]
                        miss = dict(zip(data.columns, [author, ref_words, ref_file, other_author, cand_words, cand_file, 0]))
                        data = data.append(miss, ignore_index=True)

        data = balance_classes(data)
        data.to_csv(path, index=False)

    else:    # Import pre-created data
        if os.path.exists(path):
            data = pd.read_csv(path)
            data.loc[:,['ref_words', 'cand_words']] = data.loc[:,['ref_words', 'cand_words']].applymap(ast.literal_eval)
        else:
            raise ValueError('Data file {} not found'.format(path))

    valid_split = int(split*len(data))
    train_data = data.iloc[:valid_split,:]
    valid_data = data.iloc[valid_split:,:]
    return train_data, valid_data


def create_test_set(complement = '../train-data/data_40.csv', replace = False):
    '''
    Create test set of texts with authors the model has not seen yet
    Inputs: 
        complement: Training data to create complement test set for
        replace: whether to replace test data or load pre-created data from CSV
    '''
    if replace:
        comp_data = pd.read_csv(complement)
        comp_authors = comp_data['author'].unique()    # authors in training data
        all_authors = os.listdir('../data/Reuters-50')
        authors = list(set(all_authors) - set(comp_authors))

        test_data, _ = create_data(num_authors = len(authors), spec_authors = authors, split = 1.)
        test_data.to_csv('../train-data/test_data.csv', index=False)

    else:
        test_data = pd.read_csv('../train-data/test_data.csv')
        test_data.loc[:,['ref_words', 'cand_words']] = test_data.loc[:,['ref_words', 'cand_words']].applymap(ast.literal_eval)
    return test_data


def generate_batch(data, batch_num, size):
    '''
    Generate batch of inputs texts and targets to feed to neural net
    '''

    subset = data.iloc[batch_num*size : batch_num*size + size,:]
    refs = np.stack(subset.loc[:,'ref_words'].values)
    candidates = np.stack(subset.loc[:,'cand_words'].values)
    targets = subset.iloc[:,-1].values.reshape(size, 1)
    return refs, candidates, targets


def get_accuracy(refs_out, candidates_out, threshold, labels):
    '''
    Compute accuracy of model
    Inputs: 
        refs_out: last layer from reference text
        candidates_out: last layer from candidates text
        threshold: max distance between mappings to be predicted of same author
        labels: ground truth
    Output: Accuracy between 0, 1
    '''

    dist = tf.sqrt(tf.reduce_sum(tf.square(refs_out - candidates_out), 1, keepdims = True))
    preds = tf.cast(tf.less(dist, threshold), tf.float32)
    score = tf.equal(preds, labels)
    acc = tf.reduce_mean(tf.cast(score, tf.float32))
    return acc


def forward_pass(embeds):
    '''
    Pass embeddings of texts through neural net architecture
    '''
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        initializer = tf.initializers.truncated_normal()
        seq_length = tf.ones([tf.shape(embeds)[0]], dtype = tf.int32) * 200

        # Bidirectional LSTM
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(512, activation = tf.nn.tanh)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(512, activation = tf.nn.tanh)
        LSTM_outs = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, embeds, sequence_length = seq_length, dtype = tf.float32)
        final_states = [out.h for out in LSTM_outs[1]]
        concat_states = tf.concat(final_states, axis = -1)

        # Feed forward final layers
        batch_norm0 = tf.layers.batch_normalization(concat_states)
        layer1 = tf.layers.dense(batch_norm0, 1024, kernel_initializer = initializer, activation = tf.nn.relu)
        batch_norm1 = tf.layers.batch_normalization(layer1)
        layer2 = tf.layers.dense(batch_norm1, 512, kernel_initializer = initializer, activation = tf.nn.relu)
        batch_norm2 = tf.layers.batch_normalization(layer2)
        outputs =  tf.layers.dense(batch_norm2, 256, kernel_initializer = initializer, activation = tf.nn.relu)
    return outputs


def train(train_data, valid_data, test_data, epochs = 25, batch_size = 128):
    '''
    Train model
    '''

    logger = configure_logger(modelname = 'lstm_large', level=10)

    graph = tf.Graph()
    with graph.as_default():
        embeddings = tf.constant(stored_embeddings, dtype = tf.float32)
        
        # Training data to be fed in during training
        train_candidates = tf.placeholder(tf.int32, shape = (None, sequence_length))
        train_candidates_embed = tf.nn.embedding_lookup(embeddings, train_candidates)
        train_refs = tf.placeholder(tf.int32, shape = (None, sequence_length))
        train_refs_embed = tf.nn.embedding_lookup(embeddings, train_refs)
        train_targets = tf.placeholder(tf.float32, shape = (None,1))

        # Valid data generated in its entirety
        generated_valid = generate_batch(valid_data, 0, len(valid_data))
        valid_candidates = tf.constant(generated_valid[0], dtype=tf.int32)
        valid_candidates_embed = tf.nn.embedding_lookup(embeddings, valid_candidates)
        valid_refs = tf.constant(generated_valid[1], dtype = tf.int32)
        valid_refs_embed = tf.nn.embedding_lookup(embeddings, valid_refs)
        valid_targets = tf.constant(generated_valid[2], dtype=tf.float32)
        
        # Test data generated in its entirety
        generated_test = generate_batch(test_data, 0, len(test_data))
        test_candidates = tf.constant(generated_test[0], dtype=tf.int32)
        test_candidates_embed = tf.nn.embedding_lookup(embeddings, test_candidates)
        test_refs = tf.constant(generated_test[1], dtype = tf.int32)
        test_refs_embed = tf.nn.embedding_lookup(embeddings, test_refs)
        test_targets = tf.constant(generated_test[2], dtype=tf.float32)

        # Sample of training set to compute accuracy for
        train_subset = train_data.sample(frac = 0.01)
        generated_train = generate_batch(train_subset, 0, len(train_subset))
        subset_train_candidates = tf.constant(generated_train[0], dtype=tf.int32)
        subset_train_candidates_embed = tf.nn.embedding_lookup(embeddings, subset_train_candidates)
        subset_train_refs = tf.constant(generated_train[1], dtype=tf.int32)
        subset_train_refs_embed = tf.nn.embedding_lookup(embeddings, subset_train_refs)
        subset_train_targets = tf.constant(generated_train[2], dtype=tf.float32)

        refs_outputs = forward_pass(train_refs_embed)
        candidates_outputs = forward_pass(train_candidates_embed)

        margin = tf.constant(5.)    # Margin to use in contrastive loss computation
        threshold = 1/2 * margin    # Threshold to use to predict classes

        d = tf.sqrt(tf.reduce_sum(tf.square(refs_outputs - candidates_outputs), 1, keepdims = True))    # Distance between mappings
        loss = tf.reduce_mean(train_targets * d + (1. - train_targets) * tf.maximum(0., margin - d))    # Contrastive loss

        learning_rate = tf.placeholder(tf.float32, shape=[])    # lr to be decayed during training

        # Use gradient clippings to prevent bad batches from hurting training
        clip = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        train_accuracy = get_accuracy(forward_pass(subset_train_refs_embed), forward_pass(subset_train_candidates_embed), threshold, subset_train_targets)
        valid_accuracy = get_accuracy(forward_pass(valid_refs_embed), forward_pass(valid_candidates_embed), threshold, valid_targets)
        test_accuracy = get_accuracy(forward_pass(test_refs_embed), forward_pass(test_candidates_embed), threshold, test_targets)

        saver = tf.train.Saver()    # To checkpoint model

    num_batches = len(train_data) // batch_size
    with tf.Session(graph = graph) as sess:
        tf.global_variables_initializer().run()
        best_acc = 0    # Best validatoin accuracy up to this point
        for epoch in range(epochs):
            train_data = train_data.sample(frac=1)    # Shuffle data at start of every epoch
            cum_l = 0
            for batch in range(num_batches):
                batch_candidates, batch_refs, batch_targets = generate_batch(train_data, batch, batch_size)
                feed_dict = {train_candidates: batch_candidates, train_refs: batch_refs, train_targets: batch_targets, learning_rate: 0.0003 + 0.001 * 0.7**epoch, clip: 0.08 + 0.22 * 0.5**epoch}
                _, l = sess.run([train_op, loss], feed_dict=feed_dict)
                cum_l += l
                if (batch + 1) % 500 == 0:
                    msg = 'Batch {} of {}. Avg loss over past 500 batches: {:0.3f}'.format(batch + 1, num_batches, cum_l/500)
                    logger.info(msg)
                    cum_l = 0

            valid_acc = valid_accuracy.eval()
            msg = 'Done epoch {}. '.format(epoch+1)
            msg += 'Valid accuracy: {:.1%} '.format(valid_acc)
            msg += 'Train accuracy: {:.1%} '.format(train_accuracy.eval())
            logger.info(msg)
            print(msg)

            # Checkpoint model if improvement in validation accuracy
            if valid_acc > best_acc:
                saver.save(sess, '../models/lstm-any-author/model')
                best_acc = valid_acc

    # Restore best model and compute accuracy on out-of-set authors
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, "../models/lstm-any-author/model")
        test_acc = test_accuracy.eval()
        logger.info('Test set accuracy: {:.1%} '.format(test_acc))


def run_model(num_authors = 40):
    print('Loading data')
    train_data, valid_data = create_data(num_authors = num_authors)
    test_data = create_test_set()
    test_data = test_data.sample(n = 2000)
    print('Data loaded')
    train(train_data, valid_data, test_data)

if __name__ == "__main__":
    run_model()
