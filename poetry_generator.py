import tensorflow as tf
import numpy as np
import helper
import sys

_, vocab_to_int, int_to_vocab = helper.load_preprocess()
seq_length = 25
load_dir = './save'

def get_tensors(loaded_graph):
    input_tensor = loaded_graph.get_tensor_by_name("input:0")
    initial_state = loaded_graph.get_tensor_by_name("initial_state:0")
    final_state = loaded_graph.get_tensor_by_name("final_state:0")
    probs = loaded_graph.get_tensor_by_name("probs:0")
    return input_tensor, initial_state, final_state, probs

def pick_5_words_poetry_word(probabilities, int_to_vocab, gen_sentences_length):
    result = ""
    if gen_sentences_length == 5 or gen_sentences_length == 17:
        result = '，'
    elif gen_sentences_length == 11 or gen_sentences_length == 23:
        result = '。'
    else:
        choose = np.random.choice(len(probabilities), size=30, p=probabilities)
        for i in range(30):
            word = int_to_vocab[choose[i]]
            if (word is not '，') and (word is not '。'):
                result = word
                break;
    return result

def pick_word(probabilities, int_to_vocab):
    return int_to_vocab[np.random.choice(len(probabilities), size=1, p=probabilities)[0]]

def generate_poetry(prim_words, gen_length, load_dir, seq_length):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
        input_text, initial_state, final_state, probs = get_tensors(loaded_graph)
        gen_sentences = list(prim_words)
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})
        for n in range(gen_length-len(gen_sentences)):
            dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])
            probabilities, prev_state = sess.run([probs, final_state],
                                                {input_text: dyn_input, initial_state: prev_state})
            pred_word = pick_5_words_poetry_word(probabilities[dyn_seq_length-1], int_to_vocab, len(gen_sentences))
            gen_sentences.append(pred_word)
        poetry = ''.join(gen_sentences)
        return poetry

def generate_five_word_poetry(prim_words):
	return generate_poetry(prim_words, 24, load_dir, 24)

if __name__ == "__main__":
	poetry = generate_five_word_poetry(sys.argv[1])
	print(poetry)