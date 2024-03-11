import pandas as pd
import numpy as np

# Function to initialize the Q-table


def init_q_table(states, actions):
    return {state: {action: 0.0 for action in actions} for state in states}


# Function to select the next action
def select_next_action(q_table, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(list(q_table[state].keys()))
    else:
        return max(q_table[state], key=q_table[state].get)
