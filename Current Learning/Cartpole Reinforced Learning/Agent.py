# %%
import numpy as np
from keras.models import clone_model
from collections import deque
import random
from keras.models import Sequential
from keras.layers import Dense

def create_dqn_model(input_shape, num_actions):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.8, epsilon=1.0, epsilon_decay=0.97, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = create_dqn_model((state_size,), action_size)
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        # Epsilon-greedy exploration strategy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values) # return action index

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample a random batch from replay memory
        minibatch = random.sample(self.memory, batch_size)

        # Update Q-values using the Bellman equation
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target


            # Train the model using the current state and updated Q-values
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Update the exploration-exploitation trade-off
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return
    
    def update_target_model(self):
        # Update the target model periodically
        self.target_model.set_weights(self.model.get_weights())
# %%
