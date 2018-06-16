import tensorflow as tf
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Data taken from Kaggle Competition
# https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017

data_df = pd.read_csv('results.csv', usecols=[1, 2, 3, 4, 5, 7, 8])
test_name_df = data_df.iloc[48373:, :].reset_index(drop=True)
to_replace = {}


def to_dict(l):
    d = {}
    for i, x in enumerate(l):
        d[x] = i
    return d


home_teams = data_df.home_team.unique()
away_teams = data_df.away_team.unique()
tournaments = data_df.tournament.unique()
neutral = data_df.neutral.unique()
countries = data_df.country.unique()
data_df["home_team"] = data_df.home_team.map(to_dict(home_teams))
data_df["away_team"] = data_df.away_team.map(to_dict(away_teams))
data_df["tournament"] = data_df.tournament.map(to_dict(tournaments))
data_df["neutral"] = data_df.neutral.map(to_dict(neutral))
a = pd.get_dummies(data_df["home_team"])
b = pd.get_dummies(data_df["away_team"])
c = pd.get_dummies(data_df["tournament"])
d = pd.get_dummies(data_df["neutral"])
data_df = pd.concat([a, b, c, d, data_df["home_score"], data_df["away_score"]], axis=1)

# data_df = data_df.replace(to_replace=to_replace, inplace=True)
train_df = data_df.iloc[:48373, :]
train_df_X = train_df.iloc[:, :-2].reset_index(drop=True)
home_score_df = train_df.ix[:, ["home_score", "away_score"]].reset_index(drop=True)["home_score"]
away_score_df = train_df.ix[:, ["home_score", "away_score"]].reset_index(drop=True)["away_score"]
train_df_y = pd.get_dummies((home_score_df > away_score_df))
print(train_df_y.shape)

test_df = data_df.iloc[48373:, :]
test_df_X = test_df.iloc[:, :-2].reset_index(drop=True)

batch_size = 100
num_input = data_df.shape[1]-2
n_hidden_1 = 70
n_hidden_2 = 100
num_output = 2

num_steps = 30000
learning_rate = 0.0005

index = 0


def generate_batch():
    global index
    if (index+batch_size) > train_df_X.shape[0]:
        x = train_df_X.iloc[index:train_df_X.shape[0], :]
        y = train_df_y.iloc[index:train_df_X.shape[0]]
        index = 0
    else:
        x = train_df_X.iloc[index:index + batch_size, :]
        y = train_df_y.iloc[index:index + batch_size]
        index = (index + batch_size) % len(train_df_X)
    return x, y


# Source: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
X = tf.placeholder(dtype=tf.float32, shape=[None, num_input])
Y = tf.placeholder(dtype=tf.float32, shape=[None, num_output])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_output]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    print("Start training RNN")
    for step in range(num_steps):
        batch_x, batch_y = generate_batch()
        # print(session.run([weights], feed_dict={X: batch_x, Y: batch_y}))
        session.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        if step % 500 == 0:
            acc = session.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            print('Accuracy at step %d: %f' % (step, acc))
            print(index)
    print("Finish training RNN!")
    print("Let predict World Cup!")
    test_df_y = pd.DataFrame(session.run(prediction, feed_dict={X: test_df_X}))
    results_df = pd.concat([test_name_df.iloc[:, 0:2], test_df_y.iloc[:, 1], test_df_y.iloc[:, 0]], axis=1)
    print(results_df.iloc[:, :])
