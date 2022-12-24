#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# 
# # Solving Atari Breakout with DQN
# 

# <img src="file://_static/img/tutorials/blackjack_AE_loop.jpg" width="650" alt="agent-environment-diagram" class="only-light">
# <img src="file://_static/img/tutorials/blackjack_AE_loop_dark.png" width="650" alt="agent-environment-diagram" class="only-dark">
# 
# In this tutorial, we’ll explore and solve the *Blackjack-v1*
# environment.
# 
# **Blackjack** is one of the most popular casino card games that is also
# infamous for being beatable under certain conditions. This version of
# the game uses an infinite deck (we draw the cards with replacement), so
# counting cards won’t be a viable strategy in our simulated game.
# Full documentation can be found at https://gymnasium.farama.org/environments/toy_text/blackjack
# 
# **Objective**: To win, your card sum should be greater than the
# dealers without exceeding 21.
# 
# **Actions**: Agents can pick between two actions:
#  - stand (0): the player takes no more cards
#  - hit (1): the player will be given another card, however the player could get over 21 and bust
# 
# **Approach**: To solve this environment by yourself, you can pick your
# favorite discrete RL algorithm. The presented solution uses *Q-learning*
# (a model-free RL algorithm).
# 
# 
# 

# ## Imports and Environment Setup
# 
# 
# 

# In[46]:


# Author: Matthew Trang
# License: MIT License

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym

import copy
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from skimage.color import rgb2grey
from skimage.transform import resize, rescale


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Let's start by creating the blackjack environment.
# Note: We are going to follow the rules from Sutton & Barto.
# Other versions of the game can be found below for you to experiment.

env = gym.make("ALE/Breakout-v5")


# .. code:: py
# 
#   # Other possible environment configurations are:
# 
# 
# 
# 

# ## Observing the environment
# 
# First of all, we call ``env.reset()`` to start an episode. This function
# resets the environment to a starting position and returns an initial
# ``observation``. We usually also set ``done = False``. This variable
# will be useful later to check if a game is terminated (i.e., the player wins or loses).
# 
# 
# 

# In[3]:


# reset the environment to get the first observation
done = False
observation, info = env.reset()

# observation = ndarray(210, 160, 3)


# Note that our observation is a 3-tuple consisting of 3 values:
# 
# -  The players current sum
# -  Value of the dealers face-up card
# -  Boolean whether the player holds a usable ace (An ace is usable if it
#    counts as 11 without busting)
# 
# 
# 

# ## Executing an action
# 
# After receiving our first observation, we are only going to use the
# ``env.step(action)`` function to interact with the environment. This
# function takes an action as input and executes it in the environment.
# Because that action changes the state of the environment, it returns
# four useful variables to us. These are:
# 
# -  ``next_state``: This is the observation that the agent will receive
#    after taking the action.
# -  ``reward``: This is the reward that the agent will receive after
#    taking the action.
# -  ``terminated``: This is a boolean variable that indicates whether or
#    not the environment has terminated.
# -  ``truncated``: This is a boolean variable that also indicates whether
#    the episode ended by early truncation, i.e., a time limit is reached.
# -  ``info``: This is a dictionary that might contain additional
#    information about the environment.
# 
# The ``next_state``, ``reward``,  ``terminated`` and ``truncated`` variables are
# self-explanatory, but the ``info`` variable requires some additional
# explanation. This variable contains a dictionary that might have some
# extra information about the environment, but in the Blackjack-v1
# environment you can ignore it. For example in Atari environments the
# info dictionary has a ``ale.lives`` key that tells us how many lives the
# agent has left. If the agent has 0 lives, then the episode is over.
# 
# Note that it is not a good idea to call ``env.render()`` in your training
# loop because rendering slows down training by a lot. Rather try to build
# an extra loop to evaluate and showcase the agent after training.
# 
# 
# 

# In[4]:


# sample a random action from all valid actions
action = env.action_space.sample()
# action=1

# execute the action in our environment and receive infos from the environment
observation, reward, terminated, truncated, info = env.step(action)

# observation = ndarray(210, 160, 3)
# reward=0.0
# terminated=False
# truncated=False
# info={'lives': 5, 'episode_frame_number': 4, 'frame_number': 4}


# Once ``terminated = True`` or ``truncated=True``, we should stop the
# current episode and begin a new one with ``env.reset()``. If you
# continue executing actions without resetting the environment, it still
# responds but the output won’t be useful for training (it might even be
# harmful if the agent learns on invalid data).
# 
# 
# 

# ## Building an agent
# 
# Let’s build a ``Q-learning agent`` to solve *Blackjack-v1*! We’ll need
# some functions for picking an action and updating the agents action
# values. To ensure that the agents explores the environment, one possible
# solution is the ``epsilon-greedy`` strategy, where we pick a random
# action with the percentage ``epsilon`` and the greedy action (currently
# valued as the best) ``1 - epsilon``.
# 
# 
# 

# In[42]:


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Layers
        self.conv1 = nn.Conv2d(
            in_channels=n_observations,
            out_channels=16,
            kernel_size=8,
            stride=4,
            padding=2
            )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
            )
        self.layer1 = nn.Linear(3200, 256) # fix this size
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent(object):

    def __init__(self, n_frames, n_actions, batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.05, eps_decay=200, learning_rate=0.001, tau=0.001):
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.learning_rate = learning_rate
        self.tau = tau # target network update rate

        self.policy_net = DQN(n_frames, n_actions).to(device)
        self.target_net = DQN(n_frames, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayMemory(10000)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.num_steps = 0
        self.cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.loss = nn.SmoothL1Loss()
        

    def get_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def clone(self):
        # try:
        #     del self.clone_model
        # except:
        #     pass
        
        # self.clone_model = copy.deepcopy(self.model)
        
        # for p in self.clone_model.parameters():
        #     p.requires_grad = False
        
        # if self.cuda:
        #     self.clone_model = self.clone_model.cuda()
        pass

    def push(self, *args):
        self.memory.push(*args)

    def retrieve(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state, action, reward, terminal, next_state = map(torch.cat, [*batch])
        return state, action, reward, terminal, next_state

    def process(self, state):
        state = rgb2grey(state[35:195, :, :])
        state = rescale(state, scale=0.5)
        state = state[np.newaxis, np.newaxis, :, :]
        return torch.tensor(state, device=self.device, dtype=torch.float)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 

    def play(self, episodes, train=False, load=False, plot=False, render=False, verbose=False):
    
        self.t = 0
        metadata = dict(episode=[], reward=[])
        
        # if load:
        #     self.load()

        try:
            progress_bar = tqdm(range(episodes), unit='episode')
            
            i = 0
            for episode in progress_bar:

                state = env.reset()
                state = self.process(state)
                
                done = False
                total_reward = 0

                while not done:

                    if render:
                        env.render()

                    while state.size()[1] < num_frames:
                        action = 1 # Fire

                        new_frame, reward, done, info = env.step(action)
                        new_frame = self.process(new_frame)

                        state = torch.cat([state, new_frame], 1)
                    
                    if train and np.random.uniform() < self.exploration_rate(self.t-burn_in):
                        action = np.random.choice(num_actions)

                    else:
                        action = self.act(state)

                    new_frame, reward, done, info = env.step(action)
                    new_frame = self.process(new_frame)

                    new_state = torch.cat([state, new_frame], 1)
                    new_state = new_state[:, 1:, :, :]
                    
                    if train:
                        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
                        action = torch.tensor([action], device=self.device, dtype=torch.long)
                        done = torch.tensor([done], device=self.device, dtype=torch.uint8)
                        
                        self.remember(state, action, reward, done, new_state)

                    state = new_state
                    total_reward += reward
                    self.t += 1
                    i += 1
                    
                    if not train:
                        time.sleep(0.1)
                    
                    if train and self.t > burn_in and i > batch_size:

                        if self.t % update_interval == 0:
                            self.update(batch_size)

                        if self.t % clone_interval == 0:
                            self.clone()

                        if self.t % save_interval == 0:
                            self.save(self.t)

                    if self.t % 1000 == 0:
                        progress_bar.set_description("t = {}".format(self.t))

                metadata['episode'].append(episode)
                metadata['reward'].append(total_reward)

                if episode % 100 == 0 and episode != 0:
                    avg_return = np.mean(metadata['reward'][-100:])
                    print("Average return (last 100 episodes): {:.2f}".format(avg_return))

                if plot:
                    plt.scatter(metadata['episode'], metadata['reward'])
                    plt.xlim(0, episodes)
                    plt.xlabel("Episode")
                    plt.ylabel("Return")
                    display.clear_output(wait=True)
                    display.display(plt.gcf())
            
            env.close()
            return metadata

        except KeyboardInterrupt:
            if train:
                print("Saving model before quitting...")
                self.save(self.t)
            
            env.close()
            return metadata
        
        


# To train the agent, we will let the agent play one episode (one complete
# game is called an episode) at a time and then update it’s Q-values after
# each episode. The agent will have to experience a lot of episodes to
# explore the environment sufficiently.
# 
# Now we should be ready to build the training loop.
# 
# 
# 

# In[44]:


# hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = 1000
final_epsilon = 0.1
tau = 0.001  # target network update rate
batch_size = 64


#def __init__(self, n_observations, n_actions, batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.05, eps_decay=200, learning_rate=0.001, tau=0.001):

agent = DQNAgent(
    n_observations=3, # 3 input channels
    n_actions=env.action_space.n, # discrete action space, use n to access size
    batch_size=batch_size,
    learning_rate=learning_rate,
    eps_start=start_epsilon,
    eps_decay=epsilon_decay,
    eps_end=final_epsilon,
)


# Great, let’s train!
# 
# Info: The current hyperparameters are set to quickly train a decent agent.
# If you want to converge to the optimal policy, try increasing
# the n_episodes by 10x and lower the learning_rate (e.g. to 0.001).
# 
# 
# 

# In[45]:


env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(torch.from_numpy(obs).to(device))
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.push(torch.from_numpy(obs).to(device), action.to(device), torch.from_numpy(next_obs).to(device), torch.tensor(reward).to(device))
        # update the agent
        agent.update()

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs


# ## Visualizing the training
# 
# 
# 

# In[9]:


rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()


# <img src="file://_static/img/tutorials/blackjack_training_plots.png">
# 
# 
# 

# ## Visualising the policy
# 
# 

# In[10]:


def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(agent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
plt.show()


# <img src="file://_static/img/tutorials/blackjack_with_usable_ace.png">
# 
# 
# 

# In[11]:


# state values & policy without usable ace (ace counts as 1)
value_grid, policy_grid = create_grids(agent, usable_ace=False)
fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
plt.show()


# <img src="file://_static/img/tutorials/blackjack_without_usable_ace.png">
# 
# It's good practice to call env.close() at the end of your script,
# so that any used resources by the environment will be closed.
# 
# 
# 

# ## Think you can do better?
# 
# 

# In[ ]:


# You can visualize the environment using the play function
# and try to win a few games.


# Hopefully this Tutorial helped you get a grip of how to interact with
# OpenAI-Gym environments and sets you on a journey to solve many more RL
# challenges.
# 
# It is recommended that you solve this environment by yourself (project
# based learning is really effective!). You can apply your favorite
# discrete RL algorithm or give Monte Carlo ES a try (covered in [Sutton &
# Barto](http://incompleteideas.net/book/the-book-2nd.html), section
# 5.3) - this way you can compare your results directly to the book.
# 
# Best of fun!
# 
# 
# 
