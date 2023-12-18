import gymnasium as gym
import numpy as np
import cv2
from agent_coordinator import Agent as Agent_Coordinator
import torch
from buffer import *
from agent_coordinator import *

def rgb_to_grayscale(obs):
    """Converts RGB observation to grayscale."""
    return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

def downsample_and_grayscale(obs, new_height, new_width):
    """Downsamples and converts the observation to grayscale."""
    obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    obs_resized = cv2.resize(obs_gray, (new_width, new_height), interpolation=cv2.INTER_AREA)  # Downsample
    return obs_resized


# Parameters

n_past = 5
height, width = 84, 84
current_sequence = [np.zeros((height, width), dtype=np.uint8) for _ in range(n_past)]
state = np.array(current_sequence)

env = gym.make("ALE/SpaceInvaders-v5")
n_actions = env.action_space.n

T_end = 5
input_dim = T_end*84**2

agent_coordinator = Agent_Coordinator(alpha=0.000001, beta=0.000001,
                        input_dims=input_dim, tau=0.001,
                        env=env, batch_size=32,
                        n_actions=n_actions)

print("Number of actions in Space Invaders:", n_actions)


n_games = 100000
best_score = env.reward_range[0]
score_history = []
for i in range(n_games):
    obs = env.reset()
    done = False

    done = False
    score = 0
    while not done:

        # action = env.action_space.sample()
        action = agent_coordinator.choose_action(state.reshape(1, -1))
        env_action = np.argmax(action)

        obs_, reward, done, _, _ = env.step(env_action)

        obs_downsampled = downsample_and_grayscale(obs_, height, width)
        current_sequence.pop(0)
        current_sequence.append(obs_downsampled)
        state_ = np.array(current_sequence)

        agent_coordinator.remember(state.reshape(1, -1),  action, reward,
                                   state_.reshape(1, -1), done)

        agent_coordinator.learn()

        state = state_

        score += reward

    if i % 100 == 0:
        agent_coordinator.save_models()

    score_history.append(score)
    avg_score = np.mean(score_history[-1000:])

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % avg_score)


