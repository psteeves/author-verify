import pandas as pd
import tensorflow as tf
import pickle
import ast
from lstm_contrastive import generate_batch, get_accuracy, forward_pass

dic = pickle.load(open('../models/dictionary', 'rb'))
stored_embeddings = pickle.load(open('../models/embeddings', 'rb'))

def test_model():
    test_data = pd.read_csv('../train-data/test_data.csv')
    test_data.loc[:,['ref_words', 'cand_words']] = test_data.loc[:,['ref_words', 'cand_words']].applymap(ast.literal_eval)

    graph = tf.Graph()
    with graph.as_default():
        embeddings = tf.constant(stored_embeddings, dtype = tf.float32)

        generated_data = generate_batch(test_data, 0, len(test_data))
        out_of_set_candidates = tf.constant(generated_data[0], dtype=tf.int32)
        out_of_set_candidates_embed = tf.nn.embedding_lookup(embeddings, out_of_set_candidates)
        out_of_set_refs = tf.constant(generated_data[1], dtype = tf.int32)
        out_of_set_refs_embed = tf.nn.embedding_lookup(embeddings, out_of_set_refs)
        out_of_set_targets = tf.constant(generated_data[2], dtype=tf.float32)

        threshold = 2.5
        out_of_set_accuracy = get_accuracy(forward_pass(out_of_set_refs_embed), forward_pass(out_of_set_candidates_embed), threshold, out_of_set_targets)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../models/lstm-any-author/model.meta')
        print('Graph imported')
        tf.train.latest_checkpoint('../models/lstm-any-author/')

        print('Session restored')
        print(tf.get_default_graph())
        #test_acc = sess.run([out_of_set_accuracy])
        print(test_acc)

if __name__ == "__main__":
    test_model()
