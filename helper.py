import os
import pickle

def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r", encoding="utf-8") as f:
        data = f.read()
        
    return data

def preprocess_and_save_data(dataset_path, create_lookup_tables):
    text = load_data(dataset_path)    
    text = list(text)
    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab), open('preprocess.p', 'wb'))
    
def load_preprocess():
    return pickle.load(open('preprocess.p', mode='rb'))

def save_params(params):
    pickle.dump(params, open('params.p', 'wb'))
    
def load_params():
    print("load params")
    pickle.load(open('params.p', mode='rb'))