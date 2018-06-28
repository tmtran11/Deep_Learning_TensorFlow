import tensorflow as tf
import pandas as pd
import numpy as np
import os
import operator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Data taken from Kaggle Competition
# https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017

data = pd.read_csv('results.csv', usecols=[1, 2, 3, 4, 5, 7, 8])
train_name_df = pd.read_csv('results.csv', usecols=[1, 2, 5, 8]).iloc[:48373, :].reset_index(drop=True)
test_name_df = data.iloc[48373:, :].reset_index(drop=True)


def to_dict(l):
    d = {}
    for i, x in enumerate(l):
        d[x] = i
    return d


def to_dict_neutral(l):
    neutral_d = {False: 0, "False": 0, True: 1, "True": 1}
    return neutral_d


def processing(data_df, train=False):
    home_teams = data_df.home_team.unique()
    away_teams = data_df.away_team.unique()
    tournaments = data_df.tournament.unique()
    neutral = data_df.neutral.unique()
    data_df["home_team"] = data_df.home_team.map(to_dict(home_teams))
    data_df["away_team"] = data_df.away_team.map(to_dict(away_teams))
    data_df["tournament"] = data_df.tournament.map(to_dict(tournaments))
    data_df["neutral"] = data_df.neutral.map(to_dict_neutral(neutral))
    c1 = pd.get_dummies(data_df["home_team"])
    c2 = pd.get_dummies(data_df["away_team"])
    c3 = pd.get_dummies(data_df["tournament"])
    c4 = pd.get_dummies(data_df["neutral"])
    if train:
        data_df = pd.concat([c1, c2, c3, c4, data_df["home_score"], data_df["away_score"]], axis=1)
    else:
        data_df = pd.concat([c1, c2, c3, c4], axis=1)
    return data_df


data = processing(data, train=True)
train_df = data.iloc[:48373, :]
train_df_X = train_df.iloc[:, :-2].reset_index(drop=True)
home_score_df = train_df.ix[:, ["home_score", "away_score"]].reset_index(drop=True)["home_score"]
away_score_df = train_df.ix[:, ["home_score", "away_score"]].reset_index(drop=True)["away_score"]
train_df_y = pd.get_dummies((home_score_df > away_score_df))

test_df = data.iloc[48373:, :]
test_df_X = test_df.iloc[:, :-2].reset_index(drop=True)

batch_size = 100
num_input = data.shape[1]-2
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
        session.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        if step % 500 == 0:
            acc = session.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            print('Accuracy at step %d: %f' % (step, acc))
            print(index)
    print("Finish training RNN!")

    print("Let predict World Cup!")

    # predict first round
    test_df_y = pd.DataFrame(session.run(prediction, feed_dict={X: test_df_X}))
    results_df = pd.concat([test_name_df.iloc[:, 0:2], test_df_y.iloc[:, 1], test_df_y.iloc[:, 0]], axis=1)
    print(results_df.iloc[:, :])

    # number of simulation
    num_sim = 1000
    # keep record of the champion in each simulation
    record = {}

    # simulate second round, quarter-final, semi-final, final, and championship
    for i in range(num_sim):
        first_round = results_df.values.tolist()
        score = {}
        a = ["Uruguay", "Russia", "Saudi Arabia", "Egypt"]
        b = ["Spain", "Portugal", "Iran", "Morocco"]
        c = ["France", "Denmark", "Peru", "Australia"]
        d = ["Croatia", "Argentina", "Nigeria", "Iceland"]
        e = ["Brazil", "Switzerland", "Serbia", "Costa Rica"]
        f = ["Mexico", "Germany", "Sweden", "Korea Republic"]
        g = ["England", "Belgium", "Tunisia", "Panama"]
        h = ["Japan", "Senegal", "Colombia", "Poland"]

        for match in first_round:
            if match[0] not in score:
                score[match[0]] = 0
            if match[1] not in score:
                score[match[1]] = 0
            tie_prob = np.random.uniform(0, 1)
            if tie_prob < 1-abs(match[2]-match[3]):
                score[match[0]] = score[match[0]] + 1
                score[match[1]] = score[match[1]] + 1
                continue
            prob = np.random.uniform(0, 1)
            if prob < match[2]:
                score[match[0]] = score[match[0]] + 3
                score[match[1]] = score[match[1]] - 1
            else:
                score[match[0]] = score[match[0]] - 1
                score[match[1]] = score[match[1]] + 3

        second_round_matches = []
        for board in [(a, b), (c, d), (e, f), (g, h)]:
            board1, board2 = board[0], board[1]
            sorted_score = dict(sorted(score.items(), key=operator.itemgetter(1), reverse=True))
            teamA = []
            teamB = []
            for team in sorted_score:
                if team in board1 and len(teamA)< 2:
                    teamA.append(team)
                elif team in board2 and len(teamB)< 2:
                    teamB.append(team)
            a1, a2, b1, b2 = teamA[0], teamA[1], teamB[0], teamB[1]
            is_neutral = "Russia" not in [a1, a2, b1, b2]
            second_round_matches.append([a1, b2, "FIFA World Cup qualification", is_neutral])
            second_round_matches.append([b1, a2, "FIFA World Cup qualification", is_neutral])

        def next_round(m):
            matches = pd.DataFrame(np.array(m).reshape(len(m), 4),
                                   columns=["home_team", "away_team", "tournament", "neutral"])
            x = processing(pd.concat([train_name_df, matches], axis=0).reset_index(drop=True)).iloc[48373:, :]

            results = pd.DataFrame(session.run(prediction, feed_dict={X: x}))
            results = pd.concat([matches.iloc[:, 0:2],
                                results.iloc[:, 1],
                                results.iloc[:, 0]], axis=1)
            this_round = results.values.tolist()
            next_round_matches = []
            for n, m in enumerate(this_round):
                p = np.random.uniform(0, 1)
                if n % 2 == 0:
                    if p < m[2]:
                        next_round_matches.append([m[0]])
                    else:
                        next_round_matches.append([m[1]])
                else:
                    if p < m[2]:
                        next_round_matches[n//2].append(m[0])
                    else:
                        next_round_matches[n//2].append(m[1])
                    is_neutral = "Russia" not in [next_round_matches[n//2][0:2]]
                    next_round_matches[n//2].append("FIFA World Cup qualification")
                    next_round_matches[n//2].append(is_neutral)
            return next_round_matches
        quarter_final = next_round(second_round_matches)
        semi_final = next_round(quarter_final)
        final = next_round(semi_final)
        champion = next_round(final)[0][0]

        if champion not in record:
            record[champion] = 1
        else:
            record[champion] = record[champion]+1

    # simulation results consists of percentages of championship for each team
    sim_results = []
    for team in record:
        sim_results.append([team, record[team]/num_sim])

    print(pd.DataFrame(np.array(sim_results).reshape(len(sim_results), 2)).iloc[:, :])