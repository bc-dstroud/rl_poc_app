import streamlit as st
import pandas as pd
import random

# Function to initialize the Q-table


def init_q_table(states, actions):
    return {state: {action: 0.0 for action in actions} for state in states}

# Function to update the Q-table


def update_q_table(q_table, state, action, reward, alpha, gamma):
    max_next_q = max(q_table[state].values())
    q_table[state][action] = q_table[state][action] + alpha * \
        (reward + gamma * max_next_q - q_table[state][action])


# Streamlit app title
st.title("Q-Learning Table for Job Match Score")


# Custom styles for the app
st.markdown("""
<style>
.custom-header {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    background-color: #4e73df;
    color: #4a4a4a;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}

.custom-list {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    background-color: #ffffff;
    border: 2px solid #4e73df;
    padding: 15px;
    border-radius: 10px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# Display the definition of State Space with enhanced styling
st.markdown("""
<div class="custom-header">
    <h2>🌟 Definition of State Space 🌟</h2>
</div>
<div class="custom-list">
    <p>Combine relevant information from various data sources to form the state space, represented as a vector:</p>
    <ul>
        <li><b>Accepted and Rejected Jobs</b>: Ratio or count from <code>candidates_df</code>.</li>
        <li><b>Encoded Skills</b>: Binary/numerical values from <code>skills_df</code>.</li>
        <li><b>Job History and Performance</b>: Data from <code>job_history_df</code>.</li>
        <li><b>Engagement Metrics</b>: Metrics from <code>activities_df</code>.</li>
        <li><b>Quality Scores</b>: Numerical scores from <code>quality_scores_df</code>.</li>
        <li><b>Likelihood of Job Acceptance</b>: Probability from <code>likelihood_to_accept_df</code>.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Display the definition of Actions with enhanced styling
st.markdown("""
<div class="custom-header">
    <h2>🚀 Definition of Actions 🚀</h2>
</div>
<div class="custom-list">
    <p>Actions are various job types, each representing a potential placement for candidates:</p>
    <ul>
        <li><b>Warehouse Associate</b>: Inventory management, packing, logistics.</li>
        <li><b>Food Service Worker</b>: Food preparation, customer service.</li>
        <li><b>Delivery Driver</b>: Driving, delivering goods, ensuring timely delivery.</li>
        <li><b>Cashier</b>: Handling transactions, customer interactions.</li>
        <li><b>Mechanic</b>: Vehicle maintenance, repair services.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar for defining states and actions
with st.sidebar:
    st.header("Define States and Actions")
    num_states = st.number_input("Number of States", min_value=1, value=2)
    states = [f"State {i+1}" for i in range(num_states)]

    actions = st.text_area("Enter actions separated by commas",
                           "Warehouse Associate, Food Service Worker, Delivery Driver, Cashier, Mechanic")
    actions = [action.strip() for action in actions.split(',')]

# Initialize the Q-table
if 'q_table' not in st.session_state or st.sidebar.button("Initialize Q-Table"):
    st.session_state.q_table = init_q_table(states, actions)

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Display the initial Q-table
st.write("Q-Table:")
st.dataframe(pd.DataFrame.from_dict(st.session_state.q_table, orient='index'))

# Button to simulate interaction and update Q-table for all state-action pairs
if st.button('Simulate Update for All State-Action Pairs'):
    for state in st.session_state.q_table:
        for action in st.session_state.q_table[state]:
            reward = random.randint(-10, 10)  # Random reward for simulation
            update_q_table(st.session_state.q_table, state,
                           action, reward, alpha, gamma)

    st.write("Updated Q-Table:")
    st.dataframe(pd.DataFrame.from_dict(
        st.session_state.q_table, orient='index'))
