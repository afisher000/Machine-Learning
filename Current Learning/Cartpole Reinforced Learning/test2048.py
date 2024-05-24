# %%
from Game2048Env import Game2048Env
import numpy as np
from keras.models import load_model

env = Game2048Env(size=4)
model = load_model('model.h5')
action_words = ['up', 'down', 'left', 'right']


# Initialize
state = env.state.reshape(1, -1)
print(state)

# Loop over moves
max_moves = 10
for j in range(max_moves):
    # Choose action using model
    action_weights = model.predict(
        state.reshape(1,-1), #(1,16)
        verbose = 0,
    )
    action = np.argmax(action_weights)

    # Make move
    print(f'Move {action_words[action]}')
    state, _, done, _ = env.step(action)
    print(state)
    if done:
        break




# %%
