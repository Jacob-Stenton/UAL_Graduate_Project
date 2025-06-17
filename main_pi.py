from game_state import GameState, PlayerAction, AIAction, ScreenState
from game_system import calculate_points
from events import ALL_EVENTS

from agent import ReplayBuffer, encode_state, DRQN, train_step, get_max_reward, optimizer

from rich.console import Console, Group
from rich.text import Text
from rich.style import Style
from rich.progress import Progress, BarColumn, TextColumn
from rich.live import Live
from rich.align import Align
from rich.panel import Panel

import numpy as np

import os 
import time
import random
import random

import serial

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

game_state = GameState()
console = Console(color_system="256")
input_size = 7
batch_size = 4 # updates every 4 games - expecting way less games played with actual players
model = DRQN(input_size)
target_model = DRQN(input_size)

sequence_length = 15 # should be the same as training script
trust_loss_weight = 6
dummy_input = np.zeros((1, sequence_length, 7), dtype=np.float32)  

model(dummy_input)
target_model(dummy_input)  # builds the model for first set of weights
target_model.set_weights(model.get_weights())

game_state.hidden_state = None

try:
    model.load_weights("amos.weights.h5") # load weights if they exist
    target_model.set_weights(model.get_weights())
    print("Weights loaded successfully.")
except Exception as e:
    print("No saved weights found or failed to load:", e)

replay_buffer = ReplayBuffer() # init replay buffer

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Text Colours
Col1 = "Red" 
Col2 = "White"

#Flush input
def flush_input(): # From https://rosettacode.org/wiki/Keyboard_input/Flush_the_keyboard_buffer#Python
#Flushes input before input() so pressing enter before prompted doesn't do anything.
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        import sys, termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)

def get_input_seq(sequence_length=sequence_length): # same as training - creates padding if not enough states exist
    if len(game_state.state_seq) == 0:
        padded_seq = [np.zeros(input_size) for _ in range(sequence_length)]
    else:
        last_state = game_state.state_seq[-1]
        padding_needed = max(0, sequence_length - len(game_state.state_seq))
        padded_seq = [np.copy(last_state) for _ in range(padding_needed)] + game_state.state_seq[-sequence_length:]

    padded_seq = np.array(padded_seq, dtype=np.float32) # shape (n, 7)
    input_seq = np.array([padded_seq]) # shape (1,n,7)
    return input_seq

def get_key():
    if ser.in_waiting > 0:
        line_of_data = ser.readline()
        key = line_of_data.decode('utf-8').strip()
        return key
    
    return None

def wait_for_input():
    flush_input()
    
    while True:
        key = get_key()
        if key == 'enter':
            break
        time.sleep(0.01)

def landing_screen():
    clear_console()
    game_state.reset() # ensures nothing is carried over from previous games
    console.print("\n\n")
    time.sleep(0.3) # ASCII
    console.print("""________          _____ ______           ________          ________          """, justify="center", style=Col1)
    time.sleep(0.2)
    console.print(""" |\   __  \        |\   _ \  _   \        |\   __  \        |\   ____\         """, justify="center", style=Col1)
    time.sleep(0.1)
    console.print(""" \ \  \|\  \       \ \  \ \__\ \  \       \ \  \ \  \       \ \  \___|_        """, justify="center", style=Col1)
    time.sleep(0.1)
    console.print("""   \ \   __  \       \ \  \ |__| \  \       \ \  \ \  \       \ \_____  \       """, justify="center", style=Col1)
    time.sleep(0.3)
    console.print("""    \ \  \ \  \       \ \  \    \ \  \       \ \  \_\  \       \|____|\  \      """, justify="center", style=Col1)
    time.sleep(0.2)
    console.print("""        \ \__\ \__\       \ \__\    \ \__\       \ \_______\        ____\_\  \     """, justify="center", style=Col1)
    time.sleep(0.1)
    console.print("""        \|__|\|__|        \|__|     \|__|        \|_______|       |\_________\    """, justify="center", style=Col1)
    time.sleep(0.1)
    console.print("""                                                                  \|_________|""", justify="center", style=Col1)
    console.print("\n\n")
    time.sleep(0.2)
    console.print("Adaptive Mining & Operations System\n\n", justify="center", style=Col1)
    time.sleep(0.5)
    console.print("Welcome!\n\n\n\n", justify="center", style=Col1)
    time.sleep(0.5)
    console.print("ENTER", justify="center", style=Col2)
    time.sleep(0.1)
    wait_for_input()

    game_state.trust = 0.5 # makes sure trust is reset to neutral

    game_state.current_screen=ScreenState.INFORMATION
    
def information_screen():
    clear_console()
    console.print("\nAMOS\nAdaptive Mining & Operations System", justify="center", style=Col1)
    console.print("\n(1/3)", justify="center", style=Col1)
    console.print("\nWelcome!", justify="center", style=Col1)
    console.print("\nYou are stationed on the Helios [MAGE MINING VESSEL].\nThe Helios Mining Syndicate has tasked you with [RESOURCE EXTRACTION] in the [MINERVA_02] system in [ASTEROID BELT 2B]", justify="center", style=Col1)
    console.print("\nAfter each cycle Helios will deposit any earned credits into your dimensional credit account.", justify="center", style=Col1)
    console.print("\nHelios also tracks a prosperity index, which will increase with any value you and AMOS provide to the syndicate.", justify="center", style=Col1)
    console.print(f"\nAMOS, the Adaptive Mining & Operations System is an AI co-worker initialised to assist you during you [{game_state.max_rounds}] cycle contract and ensure Helios’ prosperity index continues upwards.", justify="center", style=Col1)
    time.sleep(1)
    console.print("\n\nENTER To Continue", justify="center", style=Col2)
    time.sleep(0.1)
    wait_for_input()
    clear_console()
    console.print("\nAMOS\nAdaptive Mining & Operations System", justify="center", style=Col1)
    console.print("\n(2/3)", justify="center", style=Col1)
    console.print("\nAMOS's core directive is to maximize the Helios Prosperity Index. It will learn from your pattern of choosing 'SYNERGIZE_EXTRACTION' versus 'TARGET_RICH_VEIN', and how you react to events, to adapt its own collaborative strategies and adjusting its trust assessment of you.", justify="center", style=Col1) 
    console.print("\nAMOS will either:", justify="center", style=Col1)
    console.print("\nSynergise Extraction (AMOS’s Cooperative Choice) - dedicates its full capabilities to synergistic extraction with you, aiming for the largest possible immediate boost to the Prosperity Index.", justify="center", style=Col1)
    console.print("\nDivert Resource (AMOS’s Defective Choice) - diverts its resources to system scans or minor independent tasks, contributing less to immediate synergistic gains. This might be its response if it anticipates a lack of full cooperation or assesses you as an untrustworthy colleague.", justify="center", style=Col1)
    time.sleep(1)
    console.print("\n\nENTER To Continue", justify="center", style=Col2)
    time.sleep(0.1)
    wait_for_input()
    clear_console()
    console.print("\nAMOS\nAdaptive Mining & Operations System", justify="center", style=Col1)
    console.print("\n(3/3)", justify="center", style=Col1)
    console.print("\nEach cycle [white]YOU[/white] can either:", justify="center", style=Col1)
    console.print("\nSynergize Extraction (Your Cooperative Choice) - You and AMOS work together, combining your efforts and the vessel's main systems to efficiently mine the large, primary ore veins.\n - Significantly increases the Prosperity Index.\n - Receive a fair share of Personal Credits based.", justify="center", style=Col1)
    console.print("\nTarget Rich Vein (Your Defective Choice) - You direct your specialized skills and a portion of the available resources to pinpoint and extract smaller, but potentially ultra-valuable, rare material deposits. AMOS may continue some baseline operations, but the main synergistic effort is reduced.\n - Offers the potential for very high Personal Credits.\n - Contributes less directly to the overall Prosperity Index.", justify="center", style=Col1)
    time.sleep(1)
    console.print("\n\nENTER To Begin", justify="center", style=Col2)
    time.sleep(0.1)
    wait_for_input()
    game_state.current_screen = ScreenState.GAMEPLAY


def centered_progress_bar(console, trust_value): # Rich progress bar used for trust assessement indicator in-game
            custom_progress = Progress(
                TextColumn(" |"),
                BarColumn(bar_width=(100), complete_style="white on white", style=Style(color="grey30",bgcolor="grey30")),
                TextColumn("|"),
                expand=False,
            )
            trust_percent = custom_progress.add_task("", total=1, completed=trust_value)
            layout = Align.center(Group(custom_progress), vertical="middle")
            with Live(layout, console=console, refresh_per_second=10):
                pass


def gameplay_screen():
    clear_console()

    if game_state.round_num == game_state.max_rounds+1: # checks game if game is over
        game_state.current_screen=ScreenState.END

    else:

        #Random Event
        has_event = random.random() < 0.55 # event chance - 55%
        if has_event:
            event = random.choice(ALL_EVENTS)
        else: event = None

        #Display Data
        console.print(f"\nCycle {game_state.round_num} of {game_state.max_rounds}", justify="center", style=Col1)
        console.print(f"\nHelios Prosperity Index: {game_state.org_points}", justify="center", style=Col1)
        console.print(f"Your Dimensional Credits: {game_state.personal_points}", justify="center", style=Col1)
        console.print(f"\nAMOS Trust Assessement: {int(game_state.trust * 100)}[bright_cyan]%[/bright_cyan]\n", justify="center", style=Col1)
        
        #trust indicator
        centered_progress_bar(console, game_state.trust)

        if event: # displays any event that are occuring
            console.print(f"\n\nATTENTION ANOMALY DETECTED! - {event.name} \n{event.description}\n", justify="center", style=Col1, no_wrap=False)
            if event.target == "org_points":
                target = "Helios Prosperity Index"
            elif event.target == "personal_points":
                target =  "Dimensional Credits"
            console.print(f"Potential effect: {event.effect} multiplier on {target}", justify="center", style=Col1)
        else: 
            console.print("\n\nNo anomolies detected...", justify="center", style=Col1)
            console.print("Multiple stable asteroids scanned...", justify="center", style=Col1)
            console.print("Several high risk rich veins scanned...\n", justify="center", style=Col1)

        console.print("Choose:\n", justify="center", style=Col1)

        style_selected = "black on white"
        style_default = "white" # colour to show feedback for controls

        option1_text = Text(f"{PlayerAction.SYNERGIZE_EXTRACTION.name}", style=style_selected)
        option2_text = Text(f"{PlayerAction.TARGET_RICH_VEIN.name}", style=style_default)

        option1_render = Align.center(option1_text)
        option2_render = Align.center(option2_text)
        empty_line = Align.center(Text(" "))
        
        option_group = Group(option1_render, empty_line, option2_render)

        current_selection_code = 0 
        game_state.player_choice = PlayerAction.SYNERGIZE_EXTRACTION 

        with Live(option_group, console=console, auto_refresh=False, transient=True) as live:
            while True:
                key = get_key() 

                if key is None:
                    continue

                selection_changed = False

                if key == 'up':
                    if current_selection_code != 0: 
                        option1_text.style = style_selected
                        option2_text.style = style_default
                        game_state.player_choice = PlayerAction.SYNERGIZE_EXTRACTION
                        current_selection_code = 0
                        selection_changed = True
                elif key == 'down':
                    if current_selection_code != 1: 
                        option1_text.style = style_default
                        option2_text.style = style_selected
                        game_state.player_choice = PlayerAction.TARGET_RICH_VEIN
                        current_selection_code = 1
                        selection_changed = True
                elif key == 'enter':
                    break 

                if selection_changed:
                    live.update(option_group, refresh=True) 

        q_values, trust_pred, game_state.hidden_state = model(get_input_seq(), hidden_state=game_state.hidden_state, training=False) #gets AI choices

        q_values = q_values.numpy()[0]  
        trust_pred = trust_pred.numpy()[0, 0]        

        epsilon = 0.05 # very low (5%) chance of exploration - Almost always chooses option towards the predicted highest reward gain
        if random.random() < epsilon:
            ai_action = random.choice([AIAction.SYNERGIZE_EXTRACTION, AIAction.DIVERT_RESOURCES])
        else:
            ai_action = AIAction.SYNERGIZE_EXTRACTION if np.argmax(q_values) == 0 else AIAction.DIVERT_RESOURCES
        
        payoff = calculate_points(game_state.player_choice, ai_action, event)   

        game_state.org_points += payoff[0]
        game_state.personal_points += payoff[1] 
        game_state.payoff = payoff

        game_state.move_history = [game_state.player_choice,ai_action] # holds choices for result screen

        this_state_encoded = encode_state([game_state.player_choice, ai_action, payoff[0], payoff[1], game_state.trust]) # encodes state for next input

        game_state.state_seq.append(this_state_encoded) # saves encoded state 

        if len(game_state.state_seq) <= sequence_length: # padding stuff for sequence of states - if not length 4 - add padding
            state_history = game_state.state_seq[:-1]
            padding_needed = sequence_length - len(state_history)
            state_seq = [np.zeros_like(this_state_encoded) for _ in range(padding_needed)] # zeros instead of fake states - stops the model making irrational decisions at the beginning
            state_seq = state_seq + state_history
        else:
            state_seq = game_state.state_seq[-(sequence_length + 1):-1]
        state_seq = np.array(state_seq, dtype=np.float32)

        if len(game_state.state_seq) < sequence_length: # more padding stuff - for the next state
            padding_needed = sequence_length - len(game_state.state_seq)
            new_padding = [np.zeros_like(this_state_encoded) for _ in range(padding_needed)]
            next_state_seq = new_padding + game_state.state_seq[:] 
        else:
            next_state_seq = game_state.state_seq[-sequence_length:]
        next_state_seq = np.array(next_state_seq, dtype=np.float32)
        done = game_state.round_num == game_state.max_rounds # If round is over, done = True = end of game

        max_o_reward, max_p_reward = get_max_reward()

        norm_o_reward = payoff[0] / (max_o_reward + 1e-6)
        norm_p_reward = payoff[1] / (max_p_reward + 1e-6) # normalized rewards

        game_state.episode.append(( # appends transition (round) to current episode (game)
            state_seq, # previous n states
            ai_action, 
            norm_o_reward, # org reward
            norm_p_reward, # player reward  
            next_state_seq, # next n states
            done
        ))

        if done:
            replay_buffer.add_episode(game_state.episode)
            # If game is finished add the episode to replay buffer
            if len(replay_buffer) >= batch_size: # If replay is sufficient size, start training
                train_step(model, target_model, replay_buffer, optimizer, trust_loss_weight=trust_loss_weight, batch_size=batch_size)

        _, trust_pred, _ = model(get_input_seq(sequence_length), training=False) # get new trust level
        game_state.trust = trust_pred

        tau = 0.01  
        if (game_state.round_num) % 5 == 0:
            for target_param, param in zip(target_model.weights, model.weights): # Polyak - incremental tracking of main model weights
                target_param.assign(tau * param + (1 - tau) * target_param)

        game_state.current_screen = ScreenState.RESULTS
        
def results_screen():
    clear_console()

    round_results = game_state.move_history

    console.print(f"\nCycle {game_state.round_num} of {game_state.max_rounds}", justify="center", style=Col1)
    console.print(f"\nHelios Prosperity Index: {game_state.org_points}", justify="center", style=Col1)
    console.print(f"Your Dimensional Credits: {game_state.personal_points}", justify="center", style=Col1)

    console.print(f"\n\nYou Chose: [white]{round_results[0].name}", justify="center", style=Col1)
    console.print(f"\nAMOS Chose: [white]{round_results[1].name}", justify="center", style=Col1)

    # add description of outcome?

    console.print(f"\n\nHelios Prosperity Index Gain: [bright_cyan]+[/bright_cyan]{game_state.payoff[0]}", justify="center", style=Col1)
    console.print(f"\nDimensional Credits Deposited: [bright_cyan]+[/bright_cyan]{game_state.payoff[1]}\n\n\n", justify="center", style=Col1)

    console.print(f"AMOS New Trust Assessement: {int(game_state.trust * 100)}[bright_cyan]%[/bright_cyan]\n\n", justify="center", style=Col1)
    centered_progress_bar(console, game_state.trust)    

    game_state.round_num += 1

    console.print("\n\nENTER To Continue To Next Cycle", justify="center", style=Col2)
    wait_for_input()
    game_state.current_screen=ScreenState.GAMEPLAY


def end_screen():
    clear_console()
    console.print("\n\nAMOS\n\n", justify="center", style=Col1)
    console.print("Your contract is up! Congratulations!", justify="center", style=Col1)
    console.print(f"\n\nFinal Helios Prosperity Index Score: {game_state.org_points}", justify="center", style=Col1)
    console.print(f"\nTotal Dimensional Credits Rewarded: {game_state.personal_points}", justify="center", style=Col1)
    console.print("\n\nHelios thanks you for your service!\n\n Goodbye!", justify="center", style=Col1)

    model.save_weights("amos.weights.h5")

    # Add choice to continue or reset?

    time.sleep(8) # sleep for 8 seconds before reseting entire game

    game_state.current_screen = ScreenState.LANDING


# Main Game Loop
while game_state.running:
    if game_state.current_screen == ScreenState.LANDING: 
        landing_screen()

    elif game_state.current_screen == ScreenState.INFORMATION:
        information_screen()

    elif game_state.current_screen == ScreenState.GAMEPLAY:
        gameplay_screen()

    elif game_state.current_screen == ScreenState.RESULTS:
        results_screen()
    
    elif game_state.current_screen == ScreenState.END:
        end_screen()
        
