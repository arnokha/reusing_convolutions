import numpy as np
import pickle
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

import gym

options = {"game":"YarsRevenge-v0",            ## Game to train policy on
           "use_all_channels":False,    ## Use 3 channels (RGB) or 1(Greyscale)
           "save_model":False,          ## Save model after training
           "save_every":10,             ## Save model every x episodes
           "n_conv_layers":2,           ## Number of layers to use (**For now, only 2 supported**)
           "n_channels_out_1":20,       ## Number of channels/filters in conv1
           "n_channels_out_2":40,       ## Number of channels/filters in conv1
           "lr":0.0005,                 ## Learning rate for training
           "batch_size":1,              ## Update policy ever x episodes
           "n_episodes":3,              ## Number of episodes to train for
           "gamma":0.99,                ## Discount factor for reward
           "device":"cpu",             ## Device to train on
           "kernel_size":3,             ## No support for varying kernel sizes yet
           "render":False,               ## Render gameplay
           "n_tests":10,
           "steps_per_test":3000
          }

options["n_channels_in"] = 3 if options["use_all_channels"] else 1  ## Number of input (color) channels

if len(sys.argv) != 4:
    print(sys.argv)
    print("Usage: python this_file game n_output_channels1 n_output_channels2")
    exit()
else:
    options["game"] = sys.argv[1]
    options["n_channels_out_1"] = int(sys.argv[2])
    options["n_channels_out_2"] = int(sys.argv[3])


env = gym.make(options["game"])
observation = env.reset()

n_dim_x_d2 = observation[::4,::4,:].shape[0] 
n_dim_y_d2 = observation[::4,::4,:].shape[1] 
if options["use_all_channels"]:
    def prepro(I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[::4,::4,:] # downsample by factor of 4
        return torch.tensor(I.reshape(1, options["n_channels_in"], n_dim_x_d2, n_dim_y_d2)).float()
else:
    def prepro(I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[::4,::4,0] # downsample by factor of 4
        return torch.tensor(I.reshape(1, 1, n_dim_x_d2, n_dim_y_d2)).float()
    
n_dim_x = prepro(observation).shape[2]
n_dim_y = prepro(observation).shape[3]

## Base: Conv1 -> Conv2 -> FC -> Softmax -> Action Probabilities
class Base_Policy_2(nn.Module):
    def __init__(self, n_out_channels_1=options["n_channels_out_1"], 
                       n_out_channels_2=options["n_channels_out_2"]):
        
        super(Base_Policy_2, self).__init__()
        self.kernel_size = options["kernel_size"]
        self.conv1 = nn.Conv2d(options["n_channels_in"], n_out_channels_1, kernel_size=self.kernel_size )
        self.conv2 = nn.Conv2d(n_out_channels_1, n_out_channels_2, kernel_size=self.kernel_size )
        self.fc = nn.Linear(n_out_channels_2 * (n_dim_x - 2*(self.kernel_size-1)) *(n_dim_y - 2*(self.kernel_size-1)), 
                            env.action_space.n, bias=False) ## Assumes kernel_size=3, Can make general later
        
        ## Training
        #self.policy_history = Variable(torch.Tensor()).to(device=options["device"]) 
        self.saved_log_probs = []
        self.rewards = []
        self.reward_history = []         # Overall reward and loss history
        self.loss_history = []

    def forward(self, x):    
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.fc(x.view(-1))
        return torch.softmax(x, dim=-1)

## REINFORCE
device = options["device"]

def select_action(state, policy):
    state = state.to(device=device)
    #state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action).to(device=device))
    return action.item()

def update_policy(policy, optimizer, retain_graph=False):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + options["gamma"] * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(device=device)
    returns = (returns - returns.mean()) / (returns.std() + 0.0001)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = sum(policy_loss)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

## Seeding
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
env.seed(0)

## Inference
policy = Base_Policy_2()
policy = policy.to(device=options["device"])
optimizer = optim.SGD(policy.parameters(), lr=options["lr"], momentum=1e-3)

batch_size = options["batch_size"]
observation = env.reset()


running_reward = None
reward_sum = 0

times = []
test_num = 0

while test_num < options["n_tests"]:
    #while(episode_number < options["n_episodes"]):
    steps=0
    episode_number = 0
    observation = env.reset()
    start = time.time()
    
    while steps < options["steps_per_test"]:
        # preprocess the observation
        curr_img = prepro(observation)
        x = curr_img.reshape(1,options["n_channels_in"],n_dim_x_d2,n_dim_y_d2)

        # forward the policy network and sample an action from the returned probability
        with torch.no_grad():
            action = select_action(torch.tensor(x).float(), policy)
            observation, reward, done, info = env.step(action)
            steps += 1

            policy.rewards.append(reward)
            reward_sum += reward

            if done: # an episode finished
                #print("Total reward for this ep({0:d}): {1:.2f}".format(episode_number, reward_sum))
                episode_number += 1
                reward_sum = 0  
                observation = env.reset() # reset env
    
    end = time.time()
    test_num += 1
    times.append(end - start)

text_file = open(options["game"].rsplit("-")[0] + "/test_base_conv_cpu_{}_{}_out.txt".format(options["n_channels_out_1"],
                                                                                                  options["n_channels_out_2"]), "w")
text_file.write("Mean time: {0:.2f}\n".format(np.mean(times)))
for i in range(options["n_tests"]):
    text_file.write("Run {0:d} - time: {1:.2f}\n".format(i+1, times[i]))
text_file.close()
