# Deep Learning using TensorFlow, with low-level, mid-level, and high level implementations

*Neural Machine Translation - Seq2Seq model using Bi-directional Recurrent Neural Network and Attention Mechanism:
https://github.com/tmtran11/Deep_Learning_TensorFlow/blob/master/NMT/NMT.py
- Follow instruction from TensorFlow's tutorial, https://www.tensorflow.org/tutorials/seq2seq

*AI agent playing Pong Game - Linear Regression and Reinforcement Learning
https://github.com/tmtran11/Deep_Learning_TensorFlow/blob/master/PONG/PongGame.py
- Following instruction from  https://www.youtube.com/watch?v=t1A3NTttvBA&t=1469s,TensorFlow, "Deep Reinforcement Learning, without a PhD (Google I/O '18)"
- The difference in two pixels frame is used as inputs in order to allow the model to have an overall observation the game while particularly put weight on moving objects, whose movements cause significant change in pixels
- Using Reinforcement Learning with discounted reward, and logistic regression with cross-entropy, to create a modified loss function.. Root-mean-square propagate optimizer is used to minimize loss function
- Labels actions used for the training is sampled directly from the agents’ output. This has proven to be able to converge.
Detect a bug within tf.multinomial, as this method return index out of range if the different between the logits is really small.

*Sentimental Analysis - Binary Vector and Deep Neural Network
https://github.com/tmtran11/Deep_Learning_TensorFlow/blob/master/SENT/SentimentAnalysis_DNN.py
- Using traditional binary vector and Deep Neural Network to classified sentimental data.
- In data preprocessing, create a vocabulary. For each sentences in the data, create a binary word vector of vocabulary’s length, whose values corresponding the indexing of words in the vocabulary. For each element in the vector, if the sentence contain the word whose index in the vocabulary equal the index of the vector’s element, the element will be equals 1, and 0 otherwise.
- Using tf.estimator to read and store input from pandas frame.
- Using tf.feature_column to mark feature columns
- Using tf.estimator to use a built-in Deep Neural Network

*Sentimental Analysis - Compacted Embedding Matrix and Recurrent Neural Network
https://github.com/tmtran11/Deep_Learning_TensorFlow/blob/master/SENT/SentimentAnalysis_RNN.py
- Using a compacted embedding matrix and Recurrent Neural Network to classified sentimental data.
- Compacted embedding matrix:
  - This embedding matrix represents relative positions between each word and others
  - In data preprocessing, creating a vocabulary.
  - Generating training data to train embedding matrix by extracting pairs of words from each sentences so that each word in a pair is d  distance away from each each other. d is a parameter.
  - Using logistic regression to train the embedding matrix of size [vocabulary’s size, embedding’s size]. Embedding Matrix is labeled as tf.Variable; therefore, it is also optimized alongside with weights and biases.
- Recurrent Neural Network:
  - Perform embedding lookup to turn sentences in a numeric, embedded vector.
  - Generating batches of given batch size, and then in each batch, pad a sequences of zeros so that every training row in a batch have same and minimum time step.
  - Stacking multiple LSTM cells of forget_bias=1.0 to create a deep, recurrent neural network
  - Using tf.nn.dynamic_rnn to iteratively feed in the input and to deal with change in number of time steps in each batch.

*FIFA World Cup 2018 Prediction - One-hot Encoder, Deep Neural Network, and Decision Tree
https://github.com/tmtran11/Deep_Learning_TensorFlow/blob/master/WC/WorldCup.py
- Using International Football results from 1872-2017 from Kaggle's data (https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017) to predict relative chance of winning of each team in each match.
  - home_team, away_team: teams’ name and whether they are at home or away
  - home_score, away_score: scores of home_team and away_team
  - tournament: type of tournaments
  - neutral: True if both team is away
- Data preprocessing:
  - In csv file, if a match is neutral, replicate that match but switch the position of home_team and away_team. This is for regularization purpose.
  - Numerizing the data, then using pd.get_dummies() to one-hot encode feature columns.
  - Using pandas’ arithmetic to creating a column indicating which team won, then using pd.get_dummies() to one-hot encode and create 2 labels columns. 
- Deep Neural Network:
  - Deep Neural Network is chosen because Deep Neural Network is able to weight different and complex combinations of feature, which is suitable to deal with the minimal number of feature in this data
  - Deep Neural Network is trained on logistic regression to predict on two classes represented by two label columns.
  - The logistic, after undergoing softmax(), is use as the prediction for World Cup 2018. The prediction represent the winning percentages of each of the two teams in each match.
- The model predict in a way that agree with relative and conventional ranking of each teams, like how a human analyzer would predict. The model’s prediction only fail when there is an unexpected twist in the matchs, like how Germany loss to Mexico in the first round.
*Prediction for first round: https://docs.google.com/document/d/1-R8D7FXCMCoWM-sw-sOpYq8SS0t6I2G9AFeVLC821zE/edit?usp=sharing
- Decision Tree is implemented to predict the percentage of chamionship. 1000 simulations is run through decision tree, which predicts high chance of championship for England and Spain.
