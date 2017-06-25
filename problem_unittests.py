import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def _print_success_message():
    print('Test Passed')

def test_create_lookup_tables(create_lookup_table):
    with tf.Graph().as_default():
        test_text = '''
        自古逢秋悲寂寥，我言秋日胜春朝。晴空一鹤排云上，便引诗情到碧宵。
        男儿何不带吴钩，收取关山五十州。请君暂上凌烟阁，若个书生万户侯。
        早梅发高树，回映楚天碧。朔吹飘夜香，繁霜滋晓白。欲为万里赠，杳杳山水隔。寒英坐销落，何用慰远客？
        孤山寺北贾亭西，水面初平云脚低。几处早莺争暖树，谁家新燕啄春泥。乱花渐欲迷人眼，浅草才能没马蹄。最爱湖东行不足，绿杨阴里白沙堤。
        闽国扬帆后，蟾蜍亏复圆。秋风吹渭水，落叶满长安。此地聚会夕，当时雷雨寒。兰桡殊未返，消息海云端。
        昔看黄菊与君别，今听玄蝉我却回。五夜飕溜枕前觉，一夜颜妆镜中来。马思边草拳毛动，雕眄青云睡眼开。天地肃清堪开望，为君扶病上高台。
        巴山楚水凄凉地，二十三年弃置身。怀旧空吟闻笛赋，到乡翻似烂柯人。沉舟侧畔千帆过，病树前头万木春。今日听君歌一曲，暂凭杯酒长精神。
        荒村带返照，落叶乱纷纷。古路无行客，寒山独见君。野桥经雨断，涧水向田分。不为怜同病，何人到白云。
        一封朝奏九重天，夕贬潮州路八千。欲为圣明除弊事，肯将衰朽惜残年。云横秦岭家何在，雪拥蓝关马不前。知汝远来应有意，好收吾骨瘴江边。
        清晨入古寺，初日照高林。竹径通幽处，禅房花木深。山光悦鸟性，潭影空人心。万籁此俱寂，但余钟磬声。'''
        
        test_text = list(test_text)
        
        vocab_to_int, int_to_vocab = create_lookup_table(test_text)
        
        assert isinstance(vocab_to_int, dict), \
            'vocab_to_int is not a dictionary'
        assert isinstance(int_to_vocab, dict), \
            'int_to_vocab is not a dictionary'
        
        assert len(vocab_to_int) == len(int_to_vocab), \
            'Length to vocab_to_int and int_to_vocab do not match.'
        
        vocab_to_int_charactor_id_set = set(vocab_to_int.values())
        int_to_vocab_charactor_id_set = set(int_to_vocab.keys())
        
        assert not (vocab_to_int_charactor_id_set - int_to_vocab_charactor_id_set), \
            'vocab_to_int and int_to_vocab do not contain the same ids.'
        
        missmatches = [(word, id , id, int_tovocab[id]) for word, id in vocab_to_int.items() if int_to_vocab[id] != word]
        
        assert not missmatches,\
            'Found {} missmatches(s).'
            
        assert len(vocab_to_int) > len(set(test_text))/2,\
            'The length of vocab seems too small.'
            
    _print_success_message()

def test_tokenize(token_lookup):
    with tf.Graph().as_default():
        symbols = set(['，', '。'])
        token_dict = token_lookup()
        
        assert isinstance(token_dict, dict), \
            'Returned type is {}.'.format(type(token_dict))
            
        key_has_spaces = [k for k in  token_dict.keys() if ' ' in k]
        val_has_spaces = [val for val in token_dict.values() if ' ' in val]
        assert not key_has_spaces, \
            'The key includes spaces'
        assert not val_has_spaces, \
            'The value includes spaces'
            
        _print_success_message()
        
        
def test_get_inputs(get_inputs):
    with tf.Graph().as_default():
        input_data, targets, lr = get_inputs()
        
        assert input_data.op.type == 'Placeholder', \
            'Input not a Placeholder.'
        assert targets.op.type == 'Placeholder', \
            'Targets not a Placeholder.'
        assert lr.op.type == 'Placeholder', \
            'Learning Rate not a Placeholder.'
            
        assert input_data.name == 'input:0', \
            'Input has bad name. Found name {}'.format(input_data.name)
        
        #Check rank
        print(input_data.get_shape())
        input_rank = 0 if input_data.get_shape() == None else len(input_data.get_shape())
        targets_rank = 0 if targets.get_shape() == None else len(targets.get_shape())
        lr_rank = 0 if lr.get_shape() == None else len(lr.get_shape())
        
        assert input_rank == 2, \
            'Input has wrong rank. Rank {} found.'.format(input_rank)
        
        assert targets_rank == 2, \
            'Targets has wrong rank. Rank {} found.'.format(targets_rank)
        assert lr_rank == 0, \
            'Learning Rate has wrong rank. Rank {} found.'.format(lr_rank)
        
    _print_success_message()

def test_get_init_cell(get_init_cell):
    with tf.Graph().as_default():
        test_batch_size_ph = tf.placeholder(tf.int32)
        test_rnn_size = 256

        cell, init_state = get_init_cell(test_batch_size_ph, test_rnn_size)

        assert isinstance(cell, tf.contrib.rnn.MultiRNNCell),\
            'Cell is wrong type. Found {} type'.format(type(cell))

        assert hasattr(init_state, 'name'), \
            'Initial state does not have the "name" attrtibute. Try using `tf.identity` to set the name.'

        assert init_state.name == 'initial_state:0', \
            'Initial state does not have the correct name. Found the name {}'.format(initial_state.name)
    _print_success_message()

def test_get_embed(get_embed):
    with tf.Graph().as_default():
        embed_shape = [50, 5, 256]
        test_input_data = tf.placeholder(tf.int32, embed_shape[:2])
        test_vocab_size = 27
        test_embed_dim = embed_shape[2]

        embed = get_embed(test_input_data, test_vocab_size, test_embed_dim)

        assert embed.shape == embed_shape, \
            'Wrong shape. Found shape {}'.format(embed.shape)
    _print_success_message()

def test_build_rnn(build_rnn):
    with tf.Graph().as_default():
        test_rnn_size = 256
        test_rnn_layer_size = 2
        test_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(test_rnn_size) for _ in range(test_rnn_layer_size)])

        test_inputs = tf.placeholder(tf.float32, [None, None, test_rnn_size])
        outputs, final_state = build_rnn(test_cell, test_inputs)

        assert hasattr(final_state, 'name'), \
            'Final state does not have the "name" attribute. Try using `tf.identity` to set the name.'
        assert final_state.name == 'final_state:0', \
            'Final state does not have the correct name. Found the name {}'.format(final_state.name)

        assert outputs.get_shape().as_list() == [None, None, test_rnn_size],\
            'Outputs has wrong shape. Found shape {}'.format(outputs.get_shape())
        assert final_state.get_shape().as_list() == [test_rnn_layer_size, 2, None, test_rnn_size], \
            'Final state wrong shape. Found shape {}'.format(final_state.get_shape())

    _print_success_message()

def test_build_nn(build_nn):
    with tf.Graph().as_default():
        test_input_data_shape = [128, 5]
        test_input_data = tf.placeholder(tf.int32, test_input_data_shape)
        test_rnn_size = 256
        test_embed_dim = 300
        test_rnn_layer_size = 2
        test_vocab_size = 27
        test_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(test_rnn_size) for _ in range(test_rnn_layer_size)])
        logits, final_state = build_nn(test_cell, test_rnn_size, test_input_data, test_vocab_size, test_embed_dim)

        assert hasattr(final_state, 'name'), \
            'Final state does not have the "name" attribute.'
        assert final_state.name == 'final_state:0', \
            'Final state does not have the correct name. '

        assert logits.get_shape().as_list() == test_input_data_shape + [test_vocab_size], \
            'Outpus has wrong shape.'

        assert final_state.get_shape().as_list() == [test_rnn_layer_size, 2, None, test_rnn_size], \
            'Final state wrong shape.'
    _print_success_message()

def test_get_batches(get_batches):
    with tf.Graph().as_default():
        test_batch_size = 128
        test_seq_length = 5
        test_int_text = list(range(1000*test_seq_length))
        batches = get_batches(test_int_text, test_batch_size, test_seq_length)

        assert isinstance(batches, np.ndarray), \
            'Batches is not a Numpy array'

        assert batches.shape == (7, 2, 128, 5), \
            'Batches returned wrong shape.'

        for x in range(batches.shape[2]):
            assert np.array_equal(batches[0,0,x], np.array(range(x * 35, x * 35 + batches.shape[3]))), \
                'Batches returned wrong contents.'
            assert np.array_equal(batches[0,1,x], np.array(range(x * 35 + 1, x * 35 + 1 + batches.shape[3]))), \
                'Batches returned wrong contents.'

        last_seq_target = (test_batch_size-1) * 35 + 31
        last_seq = np.array(range(last_seq_target ,last_seq_target + batches.shape[3]))
        last_seq[-1] = batches[0,0,0,0]
        assert np.array_equal(batches[-1,1,-1], last_seq), \
            'The last target of the last batch should be the first input of the first batch.'

    _print_success_message()

def test_get_tensors(get_tensors):
    test_graph = tf.Graph()
    with test_graph.as_default():
        test_input = tf.placeholder(tf.int32, name='input')
        test_initial_state = tf.placeholder(tf.int32, name='initial_state')
        test_final_state = tf.placeholder(tf.int32, name='final_state')
        test_probs = tf.placeholder(tf.float32, name='probs')

    input_text, initial_state, final_state, probs = get_tensors(test_graph)

    assert input_text == test_input,\
        'Test input is wrong tensor'
    assert initial_state == test_initial_state, \
        'Initial state is wrong tensor'
    assert final_state == test_final_state, \
        'Final state is wrong tensor'
    assert probs == test_probs, \
        'Probabilities is wrong tensor'

    _print_success_message()

def test_pick_word(pick_word):
    with tf.Graph().as_default():
        test_probabilities = np.array([0.1, 0.8, 0.05, 0.05])
        test_int_to_vocab = {word_i : word for word_i, word in enumerate(['古','路','无','客'])}
        pred_word = pick_word(test_probabilities, test_int_to_vocab)
        assert isinstance(pred_word, str),\
            'Predicted word is wrong type.'

        assert pred_word in test_int_to_vocab.values(),\
            'Predicted word not found in int_to_vocab.'

    _print_success_message()

def test_pad_poetry_batch(pad_poetry_batch):
    vocab_to_int = {"a":0, "b":1, "c":2, "d":3, "e":4}
    pad_batch = pad_poetry_batch(["abc", "de"], 5, vocab_to_int)
    assert pad_batch == [[0, 1, 2], [3, 4, 5]],\
        'Test pad poetry batch is wrong'
        
    _print_success_message()
    
def test_get_pad_poetry_batches(get_batches):
    vocab_to_int = {"a":0, "b":1, "c":2, "d":3, "e":4}
    batch_size = 2
    pad_id = 5
    eos_id = 6
    text = "abc\ncba\nde\ncdea"
    x_b, y_b, x_bl, y_bl = get_batches(text, batch_size, vocab_to_int, pad_id, eos_id)