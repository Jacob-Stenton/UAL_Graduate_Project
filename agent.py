import numpy as np
import random
from collections import deque
import keras
import tensorflow as tf
from game_state import PlayerAction, AIAction
from game_system import points_matrix
from events import ALL_EVENTS

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

action_lookup = {
    PlayerAction.SYNERGIZE_EXTRACTION: [1,0], # for converting actions to one hot vectors
    PlayerAction.TARGET_RICH_VEIN: [0,1],
    AIAction.SYNERGIZE_EXTRACTION: [1,0],
    AIAction.DIVERT_RESOURCES: [0,1],
}

def get_max_reward():

    max_o_multiplier = max([event.effect for event in ALL_EVENTS if event.target == "org_points"], default=1.0) # gets the maximum possible multiplier that can effect org points (o rewards)
    max_p_multiplier = max([event.effect for event in ALL_EVENTS if event.target == "personal_points"], default=1.0) # the same but for personal points (p rewards)

    all_o_points = [points_matrix[i][0] * max_o_multiplier for i in points_matrix] # all possible points
    all_p_points = [points_matrix[i][1] * max_p_multiplier for i in points_matrix]

    return (np.max(all_o_points), np.max(all_p_points)) # maximum points possible for both o and p rewards

max_reward = get_max_reward()

def encode_state(state):
        encoded_state = [
            float(action_lookup[state[0]][0]), # player action one hot [cooperate, defect]
            float(action_lookup[state[0]][1]), 

            float(action_lookup[state[1]][0]), # ai action one hot [cooperate, defect]
            float(action_lookup[state[1]][1]),

            float(state[2] / (max_reward[0] + 1e-6)),# reward (org points gain) (+ 1e-6 to avoid / by 0)
            float(state[3] / (max_reward[1] + 1e-6)), # reward (personla points)

            float(state[4]) # trust level
            ]
        
        # print(encoded_state)
        return encoded_state

class ReplayBuffer:
    def __init__(self, max_size=10000, seq_len=5):
        self.buffer = deque(maxlen=max_size) # deque - when max size is reach, appending also deletes the first index
        self.seq_len = seq_len

    def __len__(self):
        return len(self.buffer)

    def add_episode(self, episode):  # list of episodes - 15 rounds of (state, actions, reward, next_state, done)
        self.buffer.append(episode)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size) # randomly picks episode from the sample for training
    
    def clear(self):
        self.buffer.clear()


def prepare_batch(replay_buffer, batch_size): # prepars a batch of (batch_size) e.g. 16 to feed into the network
    episodes = replay_buffer.sample(batch_size) # gets sample

    state_seqs = []
    actions = []
    mc_returns = []
    o_rewards = []
    p_rewards = []
    next_state_seqs = []
    dones = []

    for episode in episodes:
        cumulative = 0.0 # cumlative rewards for the entire episode (Monte Carlo)
        returns = []
        for _, _, o_reward, _, _, _ in reversed(episode): # steps back through the episodes rewards
            cumulative += o_reward
            returns.insert(0, cumulative)

        for i, transition in enumerate(episode): # steps through transitions in episodes (state, action, reward, next_state, done)
            state_seq, action, o_reward, p_reward, next_state_seq, done = transition
            state_seqs.append(state_seq)
            actions.append(0 if action == AIAction.SYNERGIZE_EXTRACTION else 1) #one hot
            mc_returns.append(returns[i])
            o_rewards.append(o_reward)
            p_rewards.append(p_reward)
            next_state_seqs.append(next_state_seq)
            dones.append(done)
    
    state_seqs = np.array(state_seqs)
    next_state_seqs = np.array(next_state_seqs)
    
    # print(f"state_seqs: {len(state_seqs)}")
    # print(f"actions: {len(actions)}")
    # print(f"o_rewards: {len(o_rewards)}")
    # print(f"p_rewards: {len(p_rewards)}")
    # print(f"next_state_seqs: {len(next_state_seqs)}")
    # print(f"dones: {len(dones)}")
    # print(f"mc_returns: {len(mc_returns)}")

    return (
        state_seqs,
        np.array(actions),
        np.array(o_rewards, dtype=np.float32),
        np.array(p_rewards, dtype=np.float32),
        next_state_seqs,
        np.array(dones, dtype=np.float32),
        np.array(mc_returns),
    )


class DRQN(keras.Model): # Actual network model
    def __init__(self, input_size, num_actions=2): # 2 actions, cooperate/defect
        super().__init__()

        self.lstm = keras.layers.LSTM(64, return_sequences=True, return_state=True)
        self.d1 = keras.layers.Dense(64, activation='relu')
        self.d2 = keras.layers.Dense(32, activation='relu')

        self.qd = keras.layers.Dense(16, activation="relu") # q-value dense
        self.td = keras.layers.Dense(16, activation="relu") # trust value dense

        self.q_values = keras.layers.Dense(num_actions) #outputs
        self.trust_level = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, hidden_state=None, training=False):
        if hidden_state is None:
            lstm_out, h, c = self.lstm(inputs, training=training) # If hidden state hasn't been created e.g. first input
        else:
            lstm_out, h, c = self.lstm(inputs, initial_state=hidden_state, training=training) # takes previous hidden state

        QvTv = self.d1(lstm_out[:, -1, :]) # q-values and trust values together - takes hidden state as input
        QvTv = self.d2(QvTv)

        Qv = self.qd(QvTv) # branched dense layer for q-values before output
        Tv = self.td(QvTv) # branched trust value dense layer
        
        q_values = self.q_values(Qv) #outputs 
        trust_level = self.trust_level(Tv)

        return q_values, trust_level, [h, c] # returns outputs & hideen + cell states


training_stats = { # for plotting
    "q_loss": [],
    "q_val_mean": [],
    "q_val_std": [],
    "target_q_mean": [],
    "target_q_std": []
}

# training step - returns loss
def train_step(model, target_model, replay_buffer, optimizer, gamma=0.95, trust_loss_weight=0.2, batch_size=15, verbose=False):
    if len(replay_buffer.buffer) < batch_size:
        return None

    state_seqs, actions, o_rewards, p_rewards, next_state_seqs, dones, mc_returns  = prepare_batch(replay_buffer, batch_size)    # gets batch for training

    with tf.GradientTape() as tape:
        q_values, trust_preds, _ = model(state_seqs, training=True) # predicts q and trust values
        q_values = tf.reduce_sum(q_values * tf.one_hot(actions, 2), axis=1) 

        target_q = mc_returns # Monte Carlo returns for q-value

        huber_loss_fn = tf.keras.losses.Huber(delta=1.0) # huber loss - less sensetive than MSE, especially with event multipliers
        q_loss = huber_loss_fn(target_q, q_values)

        trust_targets = o_rewards / (o_rewards + p_rewards + 1e-6) # Ratio between o_rewards and p_rewards - indicator how player decision effects should effect trust
        trust_targets = tf.sigmoid(7 * (trust_targets - 0.5)) # k - sensitivity of the ratio
        trust_targets = tf.expand_dims(trust_targets, axis=1)
        trust_targets = tf.cast(trust_targets, dtype=tf.float32) # avoid float 64 error
 
        trust_loss = tf.reduce_mean(tf.square(trust_preds - trust_targets)) # MSE of trust prediction and trust targets
        total_loss = q_loss + (trust_loss_weight * trust_loss) # Create total loss for back prop, trust is * by weight.

        if verbose:
            training_stats["q_loss"].append(q_loss.numpy()) # for plotting
            training_stats["q_val_mean"].append(tf.reduce_mean(q_values).numpy())
            training_stats["q_val_std"].append(tf.math.reduce_std(q_values).numpy())
            training_stats["target_q_mean"].append(tf.reduce_mean(target_q).numpy())
            training_stats["target_q_std"].append(tf.math.reduce_std(target_q).numpy())

            print(f"Q Loss: {q_loss.numpy():.4f} | "
                    f"Q-mean: {tf.reduce_mean(q_values).numpy():.2f}, Q-std: {tf.math.reduce_std(q_values).numpy():.2f} | "
                    f"TQ-mean: {tf.reduce_mean(target_q).numpy():.2f}, TQ-std: {tf.math.reduce_std(target_q).numpy():.2f} | "
                    f"Trust Loss: {trust_loss.numpy():.4f} | "
                    f"Trust-mean: {tf.reduce_mean(trust_targets).numpy():.2f}, Trust-std: {tf.math.reduce_std(trust_targets).numpy():.2f}\n ")

    grads = tape.gradient(total_loss, model.trainable_variables) # gradient decent
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) # back prop

    return total_loss.numpy()