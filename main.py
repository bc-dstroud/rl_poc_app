import pandas as pd
import numpy as np

# Function to initialize the Q-table


def init_q_table(states, actions):
    return {state: {action: 0.0 for action in actions} for state in states}
