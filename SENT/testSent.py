import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def vocabulary(df):
    s = set()
    for r in range(df.shape[0]):
        for w in df.iloc[r, 0].split():
            s.add(w)
        df.iloc[r, 0] = df.iloc[r, 0]
    return list(s)


def lower_case(df):
    for r in range(df.shape[0]):
        df.iloc[r, 0] = df.iloc[r, 0].lower()
    return df


def text_embedding(df):
    s = vocabulary(df)
    d = {}
    for i in range(len(s)):
        d[str(i)] = []
    for r in range(df.shape[0]):
        l = df.iloc[r, 0].split()
        for i, w in enumerate(s):
            if w in l:
                d[str(i)].append(1)
            else:
                d[str(i)].append(0)
    dfn = pd.DataFrame(d)
    dfn['emoticon'] = df['emoticon']
    return dfn


def lc(s):
    data_df = pd.read_csv('data.csv', usecols=[0, 1], names=["sentence", "emoticon"])
    data_df = lower_case(data_df)
    data_df = text_embedding(data_df)
    data_df = data_df.sample(frac=1)
    train_df = data_df.iloc[0:s // 10 * 8, :]
    test_df = data_df.iloc[s // 10 * 8:s, :]

    # DNN BODY
    # Training set

    # not train_df, train_df[0:-1.:]
    train_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["emoticon"], num_epochs=None, shuffle=True)
    # Prediction on the whole training set.
    predict_train_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["emoticon"], shuffle=False)
    # Prediction on the test set.
    predict_test_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df["emoticon"], shuffle=False)

    embedding_columns = []
    for i in range(data_df.shape[1]-1):
        embedding_column = tf.feature_column.numeric_column(key=str(i))
        embedding_columns.append(embedding_column)

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[100, 200],
        feature_columns=embedding_columns,
        n_classes=7,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    estimator.train(input_fn=train_fn, steps=200)
    print("training done")

    train_eval_result = estimator.evaluate(input_fn=predict_train_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_fn)

    return train_eval_result["accuracy"], test_eval_result["accuracy"]


# Plot learning curve between training set and test set.
# plot should show high variance --> need more training data
number_of_sample = [25, 50, 75, 90, 95]
test_set_error = []
training_set_error = []
for sample in number_of_sample:
    train, test = lc(sample)
    training_set_error.append(1.0-train)
    test_set_error.append(1.0-test)

plt.plot(number_of_sample, training_set_error, 'r--')
plt.plot(number_of_sample, test_set_error, 'b--')
plt.show()