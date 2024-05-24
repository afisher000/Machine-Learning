# %%
from Game2048Env import Game2048Env
from Agent import DQNAgent
import numpy as np

env = Game2048Env(size=4)
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n

# Initialize the DQNAgent
agent = DQNAgent(state_size, action_size)

# Training loop
num_episodes = 100
batch_size = 32
max_steps_per_episode = 10

for episode in range(num_episodes):
    state = env.reset().reshape(1,-1) 

    for step in range(max_steps_per_episode):
        # print(f'Step = {step}')
        # Choose action using the epsilon-greedy policy
        action = agent.act(state)

        # Take the chosen action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)
        
        # Remember the experience in the replay memory
        agent.remember(state, action, reward, next_state.reshape(1,-1), done)

        # Update the target model and train the agent
        agent.update_target_model()
        agent.replay(batch_size)

        state = next_state.reshape(1,-1)

        if done:
            break

    # Print the total reward for the episode (optional)
    total_reward = sum(experience[1] for experience in agent.memory)
    print(f"Episode {episode+1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon}, Gamma = {agent.gamma}")

# %%
