import gym
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

# bug: tf.multinomial have bug


# input: shape (210, 160, 3)
# output: shape (5600)
def read_pixels(state):
    final_state1 = np.empty((70, 160))
    final_state2 = np.empty((70, 80))

    state = np.array(state).mean(axis=2)
    state = np.reshape(state, (210, 160))

    for i in range(0, state.shape[0] - 3, 3):
        final_state1[i // 3, :] = state[i:i + 3, :].sum(axis=0)

    for i in range(0, final_state1.shape[1] - 2, 2):
        final_state2[:, i // 2] = final_state1[:, i:i + 2].sum(axis=1)

    final_state2 = final_state2.ravel()

    return final_state2.tolist()


def discount_rewards(rew, dis):
    current_reward = 0
    for x in range(len(rew) - 1, -1, -1):
        if not (rew[x] == 1 or rew[x] == -1):
            rew[x] = current_reward * dis
        current_reward = rew[x]
    return rew


def normalize_rewards(rew):
    rew = np.asarray(rew)
    mean = np.mean(rew)
    std = np.std(rew)
    new_rew = (rew - mean) / std
    return new_rew.tolist()


lr = 0.001
decay = 0.99
discount = 0.95
mid_layer = 200
iteration = 10
batch_size = 6000

obs = tf.placeholder(shape=[None, 5600], dtype=tf.float32)  # hold pixels different
act = tf.placeholder(shape=[None], dtype=tf.int32)  # hold action taken
rew = tf.placeholder(shape=[None], dtype=tf.float32)  # hold reward

Y = tf.layers.dense(obs, 200, activation=tf.nn.relu)
Ylogits = tf.layers.dense(Y, 6)
print(Ylogits.get_shape())

sample_op = tf.multinomial(logits=Ylogits, num_samples=1)

cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(act, 6), logits=Ylogits)
loss = tf.reduce_sum(rew * cross_entropy)

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.99)
train_op = optimizer.minimize(loss)

ss = tf.Session()
init = tf.global_variables_initializer()
ss.run(init)

pong_sim = gym.make('Pong-v0')


for i in range(5):
    observations = []
    actions = []
    rewards = []
    processed_rewards = []
    while len(observations) < batch_size:
        game_state = pong_sim.reset()
        previous_pix = read_pixels(game_state)
        done = False
        print("training")
        while not done:
            # pong_sim.render()
            current_pix = read_pixels(game_state)
            observation = (np.asarray(current_pix) - np.asarray(previous_pix)).tolist()
            previous_pix = current_pix
            action = ss.run(sample_op, feed_dict={obs: [observation]})
            # bug: action 6 out of range,  is returned
            if action[0][0] == 6:
                action[0][0] = np.random.randint(0, 5)
            game_state, reward, done, info = pong_sim.step(action[0][0])
            observation = read_pixels(game_state)
            observations.append(observation)
            actions.append(action[0][0])
            rewards.append(reward)
            # bug: game stop at one episode

        # check if escape loop
        processed_rewards = discount_rewards(rewards, discount)
        processed_rewards = normalize_rewards(processed_rewards)
    print("start train")
    ss.run(train_op, feed_dict={obs: observations, act: actions, rew: processed_rewards})

    print("start test")
    game_state = pong_sim.reset()
    previous_pix = read_pixels(game_state)
    done = False
    while not done:
        pong_sim.render()
        current_pix = read_pixels(game_state)
        observation = (np.asarray(current_pix) - np.asarray(previous_pix)).tolist()
        previous_pix = current_pix
        action = ss.run(sample_op, feed_dict={obs: [observation]})
        # bug: action 6 out of range,  is returned
        if action[0][0] == 6:
            print("bug")
            action[0][0] = np.random.randint(0, 5)
        game_state, reward, done, info = pong_sim.step(action[0][0])

ss.close()
