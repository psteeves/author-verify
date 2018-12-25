import tensorflow as tf
import numpy as np
import collections
import os
import pickle
import utils
from nltk.tokenize import RegexpTokenizer
w_tokenizer = RegexpTokenizer('\w+')



def create_data(words, vocab_size):
    """
    Create data of all words to train embeddings
    Input:
        words: list of all words
        vocab_size: size of desired vocabulary (least frequent words to be treated as unknown)
    Returns: dict mapping words to ints, dict mapping ints back to words, list of words converted to ints, and list of word counts
    """

    word_count = collections.Counter(words).most_common(vocab_size - 1)
    word_index_map = {'UNK': 0}    # Dictionary mapping words to indices

    for w, _ in word_count:
        word_index_map[w] = len(word_index_map)

    data = []    # All word indices in order
    unk_count = 0

    for word in words:
        if word in word_index_map:
            index = word_index_map[word]
        else:
            index = 0    # UNK
            unk_count += 1    # Update UNK count
        data.append(index)

    word_count = [('UNK', unk_count)] + word_count
    index_word_map = dict(zip(word_index_map.values(), word_index_map.keys()))

    return word_index_map, index_word_map, data, word_count


def generate_batch(data, batch_num, size, window):
    """
    Function to iteratively generate batches
    Input:
        data: list of all words to generate batches from
        batch_num: position flag to mark current batch
        size: batch size
        window: qty of words on each side of target to consider as input
    Returns: array of features of shape (size, 2*window), array of targets of shape (size, 1)
    """

    span = size + 2 * window
    subset = data[batch_num*span : (batch_num+1)*span]    # Words used to generate batch
    feats = []
    labels = []
    for i in range(size):
        left_wind = [subset[j] for j in range(i, window + i)]    # Words to the left of target
        right_wind = [subset[k] for k in range(window + i + 1, 2*window + i + 1)]    # Words to the right of target
        feats.append(left_wind + right_wind)    # Context words
        labels.append([subset[window + i]])    # Target word

    return np.array(feats), np.array(labels)


def training_loop(data, vocab_size):
    '''
    Train word2vec model
    '''

    logger = utils.configure_logger(modelname = 'word2vec')
    window_size = 3
    embedding_size = 128
    num_valid = 8    # Num words to use for validation

    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape = (None, 2*window_size))
        train_labels = tf.placeholder(tf.int32, shape = (None, 1))

        valid_inputs = tf.constant(np.random.choice(np.arange(200)[1:], num_valid, replace = False), dtype = tf.int32)    # Choose vlidation set out of most frequent words except UNK

        embeddings = tf.Variable(tf.random_uniform((vocab_size, embedding_size), -1.0, 1.0))
        embeds = tf.nn.embedding_lookup(embeddings, train_inputs)
        embed_context = tf.reduce_mean(embeds, 1)

        soft_weights = tf.Variable(tf.truncated_normal((vocab_size, embedding_size)))
        soft_biases = tf.Variable(tf.zeros((vocab_size)))

        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(soft_weights, soft_biases, train_labels, embed_context, 100, vocab_size))
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims = True))    # vector norms of embeddings
        norm_embeds = embeddings / norm    # Normalization
        valid_embeds = tf.nn.embedding_lookup(norm_embeds, valid_inputs)
        sim = tf.matmul(valid_embeds, tf.transpose(norm_embeds))    # Similarity between validation set and other embeddings


    epochs = 25
    batch_size = 512
    num_batches = len(data) // (2*window_size + batch_size)
    num_near = 5    # When validating, number of nearest neighbors to display
    with tf.Session(graph = graph) as sess:
        tf.global_variables_initializer().run()
        cum_loss = 0
        for epoch in range(epochs):
            for batch in range(num_batches):
                batch_inputs, batch_labels = generate_batch(data, batch, batch_size, window_size)
                feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
                _, l = sess.run([optimizer, loss], feed_dict=feed_dict)

                cum_loss += l
                if (batch + 1) % 1000 == 0:
                    logger.info('Batch {} of {}. Average loss over past 1000 batches: {:0.3f}'.format(batch + 1, num_batches, cum_loss/1000))
                    cum_loss = 0

            # Look at validation words and their nearest neighbors in the embedding manifold
            logger.info('\nDone epoch {}. Some common words and their neighbours:'.format(epoch))
            similarity = sim.eval()
            msg = ''
            for i in range(num_valid):
                valid_word = valid_inputs.eval()[i]
                nearest = (-similarity[i,:]).argsort()[1:num_near+1]
                msg += '\n{}: {}'.format(index_word_map[valid_word], [index_word_map[w] for w in nearest])
            logger.info(msg)

        pickle.dump(embeddings.eval(), open('../models/embeddings','wb'))


if __name__ == "__main__":
    words = w_tokenizer.tokenize(utils.get_all_words())
    word_index_map, _, data, _ = create_data(words, vocab_size = 30000)
    pickle.dump(word_index_map, open('../models/dictionary', 'wb'))
    training_loop(data, vocab_size)
