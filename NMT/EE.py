import tensorflow as tf
import numpy as np
import math
import collections
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_index = 0


class Table:
    def __init__(self, file_name):
        self.d = {}
        file = open(file_name, encoding="utf8")
        index = 0
        self.keys = []
        self.values = []
        d = {}
        self.data = []
        for line in file:
            for word in line.split():
                if word.lower() not in d:
                    d[word.lower()] = index
                    self.keys.append(word[0].lower()+word[1:])
                    self.keys.append(word[0].upper()+word[1:])
                    self.values.append(index)
                    self.values.append(index)
                    index = index + 1
            self.data.append([d[word.lower()] for word in line.split()])
        self.keys.append("</s>")
        self.values.append(index)
        self.vocab_size = index + 1


def generate_batch_EB(data, distance, batch_size):
    global data_index
    batch = []
    labels = []
    window = collections.deque(maxlen=distance * 2 + 1)
    dt = distance
    if distance > len(data[data_index]) - 1:
        dt = len(data[data_index]) - 1
    for i in range(dt + 1):
        window.append(data[data_index][i])
    while len(batch) < batch_size:
        sentence = data[data_index]
        for i in range(len(sentence)):
            if len(window) < 2:
                break
            labels.append(sentence[i])
            batch_input = sentence[i]
            while batch_input == sentence[i]:
                batch_input = window[np.random.randint(0, len(window))]
            batch.append(batch_input)
            window.append(sentence[i])
        data_index = (data_index + 1) % len(data)
        window = collections.deque(maxlen=distance * 2 + 1)
        if distance > len(data[data_index]) - 1:
            dt = len(data[data_index]) - 1
        for i in range(dt + 1):
            window.append(data[data_index][i])
    labels_and_batch = np.c_[labels, batch]
    np.random.shuffle(labels_and_batch)
    labels_and_batch = labels_and_batch[0:batch_size, :]
    labels = labels_and_batch[:, 0].reshape(batch_size, 1)
    batch = labels_and_batch[:, 1]
    return batch, labels


def embedding(file, embedding_size, distance=1, batch_size_EB=2, num_steps=100):
    table = Table(file)
    vocabulary_size = table.vocab_size
    graph = tf.Graph()

    with graph.as_default():
        # train_data is int vectors
        train_data = tf.placeholder(dtype=tf.int32, shape=[batch_size_EB])
        train_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size_EB, 1])
        embeddings = tf.Variable(tf.random_uniform(shape=[vocabulary_size, embedding_size], minval=-1, maxval=1))
        softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                          stddev=1.0 / math.sqrt(embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
        embed = tf.nn.embedding_lookup(embeddings, train_data)

        # skip-gram loss function
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                       labels=train_labels, num_sampled=1, num_classes=vocabulary_size))
        global_step = tf.Variable(10, dtype=tf.int64, trainable=True, name='global_step')
        optimizer = tf.train.AdagradDAOptimizer(1.0, global_step=global_step).minimize(loss)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Start training word2vec")
        average_loss = 0
        for step in range(num_steps):
            data = table.data
            batch_data, batch_labels = generate_batch_EB(data, distance, batch_size_EB)
            feed_dict = {train_data: batch_data, train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 20 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0
        # take out the trained embedding matrix
        embedding_matrix = session.run([embeddings])[0].tolist()
    embedding_matrix.append(np.random.random_sample((embedding_size,)).tolist())
    return embedding_matrix, table
