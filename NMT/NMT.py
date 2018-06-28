"""
@article{luong17,
  author  = {Minh{-}Thang Luong and Eugene Brevdo and Rui Zhao},
  title   = {Neural Machine Translation (seq2seq) Tutorial},
  journal = {https://github.com/tensorflow/nmt},
  year    = {2017},
}
"""

import tensorflow as tf
import itertools
from EE import embedding

train_graph = tf.Graph()
train_batch_size = 1
# train_batch_size = 40
infer_batch_size = 1
eval_batch_size = 8
embedding_size = 256
num_units = 256

learning_rate = 0.005

train_source_file, train_target_file = "eval.en.txt", "eval.vi.txt"
eval_source_file, eval_target_file = "eval.en.txt", "eval.vi.txt"
# train_source_file, train_target_file = "train.en.txt", "train.vi.txt"
# eval_source_file, eval_target_file = "eval.en.txt", "eval.vi.txt"
source_embedding_matrix, source_table = embedding(train_source_file, embedding_size)
target_embedding_matrix, target_table = embedding(train_target_file, embedding_size)

src_vocab_size = source_table.vocab_size
tg_vocab_size = target_table.vocab_size


def processing(table, file):
    keys = tf.constant(table.keys)
    values = tf.constant(table.values)
    tf_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
    dataset = tf.data.TextLineDataset(file)
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    dataset = dataset.map(lambda words: (words, tf.size(words)))
    dataset = dataset.map(lambda words, size: (tf_table.lookup(words), size))
    return dataset


def infer_processing(table, dataset):
    keys = tf.constant(table.keys)
    values = tf.constant(table.values)
    tf_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    dataset = dataset.map(lambda words: (words, tf.size(words)))
    dataset = dataset.map(lambda words, size: (tf_table.lookup(words), size))
    return dataset


class BuildTrainModel:
    def __init__(self, batched_iterator):
        ((source, source_lengths), (target, target_lengths)) = batched_iterator.get_next()

        # Embedding
        embedding_encoder = tf.constant(source_embedding_matrix)
        encoder_inputs = tf.nn.embedding_lookup(embedding_encoder, source)
        embedding_decoder = tf.constant(target_embedding_matrix)
        decoder_inputs = tf.nn.embedding_lookup(embedding_decoder, target)

        decoder_outputs = tf.slice(tf.concat([target, tf.constant([[tg_vocab_size-1]])], 1), [0, 1], tf.shape(target))
        # decoder_outputs = tf.nn.embedding_lookup(embedding_decoder, decoder_outputs)

        # Bi-directional encoder-cell
        forward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        backward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
            forward_cell, backward_cell, encoder_inputs,
            sequence_length=source_lengths, time_major=True, dtype=tf.float32)

        encoder_outputs = tf.concat(bi_outputs, -1)

        # attention_states: [batch_size, max_time, num_units]
        attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, attention_states,
            memory_sequence_length=source_lengths)

        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                           attention_layer_size=num_units)

        projection_layer = tf.layers.Dense(tg_vocab_size, use_bias=False)
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, target_lengths, time_major=True)

        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                                  initial_state=decoder_cell.zero_state(dtype=tf.float32,
                                                                                        batch_size=train_batch_size),
                                                  output_layer=projection_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)
        logits = outputs.rnn_output

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
        target_weights = tf.cast(tf.sequence_mask(target_lengths), dtype=tf.float32)

        self.loss = (tf.reduce_sum(tf.matmul(cross_entropy, target_weights)) / train_batch_size)

        # clip-gradient
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params))
        self.tables_initializer = tf.tables_initializer()
        self.saver = tf.train.Saver()

    def train(self, session):
        session.run(self.tables_initializer)
        train_loss, _ = session.run([self.loss, self.train_op])
        return train_loss


class BuildEvalModel:
    def __init__(self, batched_iterator):
        ((source, source_lengths), (target, target_lengths)) = batched_iterator.get_next()

        # Embedding
        embedding_encoder = tf.constant(source_embedding_matrix)
        encoder_inputs = tf.nn.embedding_lookup(embedding_encoder, source)
        embedding_decoder = tf.constant(target_embedding_matrix)
        decoder_inputs = tf.nn.embedding_lookup(embedding_decoder, target)
        decoder_outputs = tf.slice(tf.concat([target, tf.constant([[tg_vocab_size - 1]])], 1), [0, 1], tf.shape(target))

        # Bi-directional encoder-cell
        forward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        backward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
            forward_cell, backward_cell, encoder_inputs,
            sequence_length=source_lengths, time_major=True, dtype=tf.float32)

        encoder_outputs = tf.concat(bi_outputs, -1)

        # attention_states: [batch_size, max_time, num_units]
        attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, attention_states,
            memory_sequence_length=source_lengths)
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                           attention_layer_size=num_units)

        projection_layer = tf.layers.Dense(tg_vocab_size, use_bias=False)
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, target_lengths, time_major=True)

        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                                  initial_state=decoder_cell.zero_state(dtype=tf.float32,
                                                                                        batch_size=eval_batch_size),
                                                  output_layer=projection_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)
        logits = outputs.rnn_output

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
        target_weights = tf.cast(tf.sequence_mask(target_lengths), dtype=tf.float32)
        self.loss = (tf.reduce_sum(tf.matmul(cross_entropy, target_weights)) / eval_batch_size)
        self.saver = tf.train.Saver()

    def eval(self, session):
        return session.run(self.loss)


class BuildInferenceModel:
    def __init__(self, batched_iterator):
        (source, source_lengths) = batched_iterator.get_next()

        embedding_encoder = tf.constant(source_embedding_matrix)
        encoder_inputs = tf.nn.embedding_lookup(embedding_encoder, source)
        embedding_decoder = tf.constant(target_embedding_matrix)

        forward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        backward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
            forward_cell, backward_cell, encoder_inputs,
            sequence_length=source_lengths, time_major=True, dtype=tf.float32)

        encoder_outputs = tf.concat(bi_outputs, -1)

        # attention_states: [batch_size, max_time, num_units]
        attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, attention_states,
            memory_sequence_length=source_lengths)
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                           attention_layer_size=num_units)

        projection_layer = tf.layers.Dense(tg_vocab_size, use_bias=False)

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_decoder,
            tf.fill(dims=[infer_batch_size], value=tf.constant(src_vocab_size-1, tf.int32)),
            tf.constant(tg_vocab_size-1, tf.int32))
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                                  initial_state=decoder_cell.zero_state(dtype=tf.float32,
                                                                                        batch_size=infer_batch_size),
                                                  output_layer=projection_layer)
        # Dynamic decoding
        maximum_iterations = tf.round(tf.reduce_max(source_lengths) * 2)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        self.translations = outputs.sample_id
        self.saver = tf.train.Saver()

    def infer(self, session):
        return session.run(self.translations)


train_graph = tf.Graph()
eval_graph = tf.Graph()
infer_graph = tf.Graph()

with train_graph.as_default():
    train_source_dataset = processing(source_table, train_source_file)
    train_target_dataset = processing(target_table, train_target_file)
    train_dataset = tf.data.Dataset.zip((train_source_dataset, train_target_dataset))

    batched_dataset = train_dataset.padded_batch(
        batch_size=train_batch_size,
        padded_shapes=((tf.TensorShape([None]),  # source vectors of unknown size
                        tf.TensorShape([])),  # size(source)
                       (tf.TensorShape([None]),  # target vectors of unknown size
                        tf.TensorShape([]))),  # size(target)
        padding_values=((tf.constant(src_vocab_size-1, tf.int32),  # source vectors padded on the right with src_eos_id
                         0),  # size(source) -- unused
                        (tf.constant(tg_vocab_size-1, tf.int32),  # target vectors padded on the right with tgt_eos_id
                         0)))

    train_iterator = batched_dataset.make_initializable_iterator()
    train_model = BuildTrainModel(train_iterator)
    initializer = tf.global_variables_initializer()

with eval_graph.as_default():
    eval_source_dataset = processing(source_table, eval_source_file)
    eval_target_dataset = processing(target_table, eval_target_file)
    eval_dataset = tf.data.Dataset.zip((eval_source_dataset, eval_target_dataset))

    eval_dataset = eval_dataset.padded_batch(
        batch_size=eval_batch_size,
        padded_shapes=((tf.TensorShape([None]),  # source vectors of unknown size
                        tf.TensorShape([])),  # size(source)
                       (tf.TensorShape([None]),  # target vectors of unknown size
                        tf.TensorShape([]))),  # size(target)
        padding_values=((tf.constant(src_vocab_size-1, tf.int32),  # source vectors padded on the right with src_eos_id
                         0),  # size(source) -- unused
                        (tf.constant(tg_vocab_size-1, tf.int32),  # target vectors padded on the right with tgt_eos_id
                         0)))

    eval_iterator = eval_dataset.make_initializable_iterator()
    eval_model = BuildEvalModel(eval_iterator)

with infer_graph.as_default():
    infer_inputs = tf.placeholder(tf.string, shape=(infer_batch_size,))
    infer_dataset = tf.data.Dataset.from_tensor_slices(infer_inputs)
    infer_dataset = infer_processing(source_table, infer_dataset)

    infer_dataset = infer_dataset.padded_batch(
        batch_size=infer_batch_size,
        padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])))

    infer_iterator = infer_dataset.make_initializable_iterator()
    infer_model = BuildInferenceModel(infer_iterator)

checkpoints_path = "/tmp/model/checkpoints"

train_sess = tf.Session(graph=train_graph)
eval_sess = tf.Session(graph=eval_graph)
infer_sess = tf.Session(graph=infer_graph)

train_sess.run(initializer)
train_sess.run(train_iterator.initializer)

EVAL_STEPS = INFER_STEPS = 100


for i in range(10):
    print(i)
    train_loss = train_model.train(train_sess)

    if i % EVAL_STEPS == 0:
        checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=i)
        eval_model.saver.restore(eval_sess, checkpoint_path)
        eval_sess.run(eval_iterator.initializer)
        for _ in range(6):
            print(" eval loss: ")
            print(eval_model.eval(eval_sess))

    if i % INFER_STEPS == 0:
        infer_input_data = ["what is your name ?",
                            "you forget something"
                            "let have breakfast together"
                            "I love climbing mountain"
                            "No, don't do that"]
        checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=i)
        infer_model.saver.restore(infer_sess, checkpoint_path)
        infer_sess.run(infer_iterator.initializer, feed_dict={infer_inputs: infer_input_data})
        for x in range(5):
            print(infer_input_data[x])
            print(infer_model.infer(infer_sess))

