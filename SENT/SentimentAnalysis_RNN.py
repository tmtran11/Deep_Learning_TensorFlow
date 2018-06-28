import tensorflow as tf
import numpy as np
import pandas as pd
import math
import collections
from SENT.preprocessing import lower_case, vocabulary, num_vec
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Train Embedding Matrix using linear regression

# process data
data_df = pd.read_csv('data.csv', usecols=[0, 1], names=["sentence", "label"])
data_df = lower_case(data_df)
data_df = data_df.sample(frac=1)
index_lookup, word_lookup, vocabulary = vocabulary(data_df)
data_df_vec = num_vec(data_df, index_lookup)


data_index = 0
batch_size_EB = 20
vocabulary_size = len(vocabulary)
embedding_size = int(vocabulary_size ** 0.25)
num_sampled = 1
num_steps = 10000
d = 2


def generate_batch_EB(distance):
    global data_index
    batch = []
    labels = []
    window = collections.deque(maxlen=distance * 2 + 1)
    dt = distance
    if distance > len(data_df_vec.iloc[data_index, 0]) - 1:
        dt = len(data_df_vec.iloc[data_index, 0]) - 1
    for i in range(dt + 1):
        window.append(data_df_vec.iloc[data_index, 0][i])
    while len(batch) < batch_size_EB:
        sentence = data_df_vec.iloc[data_index, 0]
        for i in range(len(sentence)):
            if len(window) < 2:
                break
            labels.append(sentence[i])
            batch_input = sentence[i]
            while batch_input == sentence[i]:
                batch_input = window[np.random.randint(0, len(window))]
            batch.append(batch_input)
            window.append(sentence[i])
        data_index = (data_index + 1) % data_df_vec.shape[0]
        window = collections.deque(maxlen=distance * 2 + 1)
        if distance > len(data_df_vec.iloc[data_index, 0]) - 1:
            dt = len(data_df_vec.iloc[data_index, 0]) - 1
        for i in range(dt + 1):
            window.append(data_df_vec.iloc[data_index, 0][i])
    labels_and_batch = np.c_[labels, batch]
    np.random.shuffle(labels_and_batch)
    labels_and_batch = labels_and_batch[0:batch_size_EB, :]
    labels = labels_and_batch[:, 0].reshape(batch_size_EB, 1)
    batch = labels_and_batch[:, 1]
    return batch, labels


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
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
    global_step = tf.Variable(10, dtype=tf.int64, trainable=True, name='global_step')
    optimizer = tf.train.AdagradDAOptimizer(1.0, global_step=global_step).minimize(loss)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Start training word2vec")
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch_EB(d)
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
    embedding_matrix = session.run([embeddings])[0]

print("Finish training word2vec!")

# Train sentiment analysis using Recurrent Neural Network
# RNN Parameters
learning_rate = 0.001
num_steps = 100000
batch_size_RNN = 20
display_step = 200

num_input = embedding_size
time_steps = 28
num_hidden = 8
num_classes = 7


def embedding(data_vec):
    e = []
    for r in range(data_vec.shape[0]):
        temp = []
        for n in data_vec.iloc[r, 0]:
            temp.append(embedding_matrix[n, :].tolist())
        e.append(temp)
    return e


data_embed_x = embedding(data_df_vec)
train_embed_x = data_embed_x[0:len(data_embed_x) // 10 * 8]
test_embed_x = data_embed_x[len(data_embed_x) // 10 * 8:]
train_embed_y = data_df_vec.iloc[0:data_df_vec.shape[0] // 10 * 8, 1].tolist()
test_embed_y = data_df_vec.iloc[data_df_vec.shape[0] // 10 * 8:, 1].tolist()

index = 0


def one_hot_encoder(list_labels):
    all_one_hot = []
    for i in list_labels:
        one_hot = []
        for n in range(num_classes):
            if n == i:
                one_hot.append(1)
            else:
                one_hot.append(0)
        all_one_hot.append(one_hot)
    return all_one_hot


def generate_batch_RNN():
    global index
    x = train_embed_x[index:index+batch_size_RNN]
    max_pad_length = max(len(r) for r in x)
    for r in range(len(x)):
        for _ in range(max_pad_length-len(x[r])):
            x[r].append([0]*embedding_size)
    y = train_embed_y[index:index+batch_size_RNN]
    index = (index+1) % len(train_embed_x)
    return x, one_hot_encoder(y)


X = tf.placeholder(dtype=tf.float32, shape=[None, None, num_input])
Y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

# weight and bias
weights = tf.Variable(tf.random_normal([num_hidden, num_classes]))
biases = tf.Variable(tf.random_normal([num_classes]))


def RNN(x, w, b):
    lstm = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, forget_bias=1.0)
    outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=x, dtype=tf.float32)
    return tf.matmul(outputs[:, -1], w)+b


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)
# prediction_shape = tf.shape(prediction)
# y_shape = tf.shape(Y)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    print("Start training RNN")
    for step in range(num_steps):
        batch_x, batch_y = generate_batch_RNN()
        session.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        # print(session.run([prediction_shape, y_shape], feed_dict={X: batch_x, Y: batch_y}))
        if step % 200 == 0:
            acc = session.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            print('Accuracy at step %d: %f' % (step, acc))
    print("Finish training RNN!")



