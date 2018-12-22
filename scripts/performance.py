from lstm_contrastive import run_model
import pickle


def compare_num_authors():
    qty = [30, 35, 40]
    results = {}
    for qt in qty:
        output = run_model(qt)
        results[qt] = output
    pickle.dump(results, open('../results/accuracies','wb'))

if __name__ == "__main__":
    compare_num_authors()
