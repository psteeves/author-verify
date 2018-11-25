from lstm_per_author import run_model
import pickle
import os

def get_all_results(length = 100, output = False):
    parent_dir = '../data/Reuters-50'
    authors = os.listdir(parent_dir)
    results = {}
    count = 0
    for author in authors:
        results[author] = run_model(author, length)
        print('Done {}/{}'.format(count, len(authors)))
        count += 1
    pickle.dump(results, open('../results/accuracies', 'wb'))
    if output:
        return results


def compare_text_lengths():
    lengths = [50, 100, 150, 200, 300, 400]
    results = {}
    for length in lenghts:
        output = get_all_results(length, output=True)
        accuracies = [output[author]['Test accuracy'] for author in output.keys()]
        results[length] = np.mean(accuracies)
    pickle.dump(results, open('../results/accuracies_lengths','wb'))

if __name__ == "__main__":
    compare_text_lengths()
