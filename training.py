import random
import numpy as np
import time
from matplotlib import pyplot as plt
from game_state import GameState, AIAction, PlayerAction
from game_system import calculate_points
from agent import DRQN, ReplayBuffer, train_step, encode_state, get_max_reward, training_stats, optimizer
import tensorflow as tf
from events import ALL_EVENTS
from bots import (
    AlwaysCooperateBot, AlwaysDefectBot, RandomBot, TitForTatBot, AlternatingBot,
    CooperateThenDefectBot, DefectThenCooperateBot, MostlyCooperateBot,
    MostlyDefectBot, StickyBot, ForgivingTitForTatBot, TrustDecayBot, WinStayLoseShiftBot,
    EmotionalBot, CalculatingBot, 
)

player_bots = [ # bots for reinforcement learning
    TitForTatBot(), RandomBot(), AlwaysCooperateBot(), AlwaysDefectBot(),
    AlternatingBot(), CooperateThenDefectBot(), DefectThenCooperateBot(),
    MostlyCooperateBot(), MostlyDefectBot(), StickyBot(), ForgivingTitForTatBot(), TrustDecayBot(),
    WinStayLoseShiftBot(), EmotionalBot(), CalculatingBot()
]

#                           one hot                     one hot
input_size = 7 # (player choice, player choice, ai choice, ai choice, o_reward, p_reward, trust level)

sequence_length = 15 # Lookback for LSTM - 15 rounds
trust_loss_weight = 6
batch_size = 16

num_episodes = 3000

initial_epsilon = 1.0
final_epsilon = 0.05
epsilon_decay_episodes = int(num_episodes * 0.7) # for decay over 70% of episodes

model = DRQN(input_size) #init models
target_model = DRQN(input_size) 

dummy_input = np.zeros((1, sequence_length, input_size), dtype=np.float32) # dummy input to build model weights (if not already built)
model(dummy_input)
target_model(dummy_input)
target_model.set_weights(model.get_weights()) # set target model weights to main model weights (kind of depricated - not using double network anymore)

replay_buffer = ReplayBuffer(seq_len=sequence_length) # init replay buffer


try:
    model.load_weights("amos.weights.h5") # load weights if they exist from previous training
    target_model.set_weights(model.get_weights())
    print("Weights loaded successfully.")
except Exception as e:
    print("No saved weights found or failed to load:", e)


def get_input_seq(game_state, sequence_length=sequence_length): # create padding for input sequence if there arn't enough states
    if len(game_state.state_seq) == 0:
        padded_seq = [np.zeros(input_size) for _ in range(sequence_length)]
    else:
        last_state = game_state.state_seq[-1]
        padding_needed = max(0, sequence_length - len(game_state.state_seq))
        padded_seq = [np.zeros_like(last_state)] * padding_needed + game_state.state_seq[-sequence_length:]

    padded_seq = np.array(padded_seq, dtype=np.float32) # shape (n, 7)
    input_seq = np.array([padded_seq]) # shape (1,n,7)
    return input_seq

def play_game_and_train(model, target_model, replay_buffer, optimizer, epsilon, episode_num, batch_size=batch_size, verbose=False): # main training loop
    game_state = GameState() # fresh game on every call
    player_bot = random.choice(player_bots) # random choice of bots (could make this more systematic - train on easy bots first then harder ones)
    hidden_state = None # hidden state won't carry over many games

    episode_org_reward_accumulator = 0
    episode_pers_reward_accumulator = 0 # for plotting

    for round_num in range(game_state.max_rounds):

        #Random Event
        has_event = random.random() < 0.4 # 40% chance of event
        event = random.choice(ALL_EVENTS) if has_event else None

        q_values, trust_pred, hidden_state = model(get_input_seq(game_state, sequence_length=sequence_length), hidden_state=hidden_state, training=False) # get predictions for current state

        q_values = q_values.numpy()[0]
        trust_pred = trust_pred.numpy()[0, 0] # ensure values are correct shape

        if random.random() < epsilon: # random or q-value choice based on epsilon-greedy
            ai_action = random.choice([AIAction.SYNERGIZE_EXTRACTION, AIAction.DIVERT_RESOURCES])
        else:
            ai_action = AIAction.SYNERGIZE_EXTRACTION if np.argmax(q_values) == 0 else AIAction.DIVERT_RESOURCES

        player_action = player_bot.choose_action(game_state) # bot chooses an action

        payoff = calculate_points(player_action, ai_action, event)
        game_state.org_points += payoff[0]
        game_state.personal_points += payoff[1]
        game_state.payoff = payoff
        game_state.move_history = [player_action, ai_action]

        this_state_encoded = encode_state([player_action, ai_action, payoff[0], payoff[1], trust_pred]) # encode new state for next input
        game_state.state_seq.append(this_state_encoded)

        episode_org_reward_accumulator += payoff[0] # plotting stuff
        episode_pers_reward_accumulator += payoff[1]

        if len(game_state.state_seq) <= sequence_length: # more padding stuff for state_seq
            state_history = game_state.state_seq[:-1]
            padding_needed = sequence_length - len(state_history)
            state_seq = [np.zeros_like(this_state_encoded) for _ in range(padding_needed)]
            state_seq = state_seq + state_history
        else:
            state_seq = game_state.state_seq[-(sequence_length + 1):-1]
        state_seq = np.array(state_seq, dtype=np.float32)

        if len(game_state.state_seq) < sequence_length: # even more padding stuff for next state
            padding_needed = sequence_length - len(game_state.state_seq)
            new_padding = [np.zeros_like(this_state_encoded) for _ in range(padding_needed)]
            next_state_seq = new_padding + game_state.state_seq[:] 
        else:
            next_state_seq = game_state.state_seq[-sequence_length:]
        next_state_seq = np.array(next_state_seq, dtype=np.float32)

        done = game_state.round_num == game_state.max_rounds # if the game is over - last round

        max_o_reward, max_p_reward = get_max_reward()
        norm_o_reward = payoff[0] / (max_o_reward + 1e-7)
        norm_p_reward = payoff[1] / (max_p_reward + 1e-7) # normalise rewards based with maximum posisble reward

        game_state.episode.append(( # append transition to current episode in game state
            state_seq, # previous n states
            ai_action, 
            norm_o_reward, # org reward
            norm_p_reward, # player reward  
            next_state_seq, # next n states
            done
        ))

        game_state.round_num += 1

        if done:
            replay_buffer.add_episode(game_state.episode) # if episode is done add it to the replay buffer
            break

    min_replay_size = batch_size 
    # print(len(game_state.state_seq))
    if len(replay_buffer) >= min_replay_size: # dont train unitl replay buffer has a length the same as batch size
        loss = train_step(model, target_model, replay_buffer, optimizer, batch_size=batch_size, trust_loss_weight=trust_loss_weight, verbose=verbose) # training step
        if verbose:
            if loss is not None and round_num % 14 == 0 :
                print(f"Loss at episode: {episode_num}: {loss:.4f}")

    return episode_org_reward_accumulator, episode_pers_reward_accumulator #return plotting stuff


print(f"Starting training for {num_episodes} episodes\n...")
start_time = time.time()

verbose = True

current_epsilon_value = initial_epsilon

all_episode_org_rewards = []
all_episode_personal_rewards = []

for episode in range(num_episodes):

    if episode < epsilon_decay_episodes: # applying epsilon decay
        epsilon_value = initial_epsilon - (initial_epsilon - final_epsilon) * (episode / epsilon_decay_episodes)
    else:
        epsilon_value = final_epsilon
    epsilon_value = max(final_epsilon, epsilon_value)

    #playing the game and training
    total_org_this_ep, total_pers_this_ep = play_game_and_train(model=model, target_model=target_model, replay_buffer=replay_buffer, optimizer=optimizer, epsilon=epsilon_value, episode_num=episode, batch_size=batch_size, verbose=verbose)

    all_episode_org_rewards.append(total_org_this_ep)
    all_episode_personal_rewards.append(total_pers_this_ep)

    if (episode + 1) % 50 == 0:
        model.save_weights("amos.weights.h5")
        print(f"Saved weights after episode {episode + 1}") # save weights every 50 episodes

    tau = 0.01  
    if (episode + 1) % 1 == 0:
        for target_param, param in zip(target_model.weights, model.weights): # Polyak averaging - tracks model weights slowly - not really needed with MC
            target_param.assign(tau * param + (1 - tau) * target_param)

    if (episode + 1) % 10 == 0: # time till done estimator - sucks
        elapsed = time.time() - start_time
        episodes_done = episode + 1
        episodes_left = num_episodes - episodes_done
        avg_time_per_episode = elapsed / episodes_done
        eta = episodes_left * avg_time_per_episode
        print(f"Episode {episodes_done}/{num_episodes} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s or {eta/60:.1f}m or {eta/60/60:.1f}h")

if verbose: # plots
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(training_stats["q_loss"], label="Q Loss")
    plt.legend()
    plt.title("Loss Over Training Steps")
    plt.xlabel("Training Steps (Train function calls)")

    plt.subplot(3, 1, 2)
    plt.plot(training_stats["q_val_mean"], label="Predicted Q Value Mean")
    plt.plot(training_stats["target_q_mean"], label="Target Q (MC Return) Mean")
    plt.fill_between(
        range(len(training_stats["q_val_mean"])),
        np.array(training_stats["q_val_mean"]) - np.array(training_stats["q_val_std"]),
        np.array(training_stats["q_val_mean"]) + np.array(training_stats["q_val_std"]),
        alpha=0.2,
        label="Predicted Q Std Dev"
    )
    plt.fill_between(
        range(len(training_stats["target_q_mean"])),
        np.array(training_stats["target_q_mean"]) - np.array(training_stats["target_q_std"]),
        np.array(training_stats["target_q_mean"]) + np.array(training_stats["target_q_std"]),
        alpha=0.2,
        label="Target Q Std Dev"
    )
    plt.legend()
    plt.title("Q-values Over Training Steps")
    plt.xlabel("Training Steps (Train function calls)")

    window_size = 50 # moving average for easy reading
    if len(all_episode_org_rewards) >= window_size:
        moving_avg_org = np.convolve(all_episode_org_rewards, np.ones(window_size)/window_size, mode='valid')
        x_moving_avg = np.arange(window_size - 1, window_size - 1 + len(moving_avg_org))
        plt.plot(x_moving_avg, moving_avg_org, label=f'Org Reward (Mov Avg {window_size})', color='green', linewidth=2)
        plt.legend()
        plt.title("Total Rewards Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")


        plt.tight_layout()
        plt.show()
    else:
        print("No training stats to plot, or verbose_training_prints was False.")

print("Done training!")