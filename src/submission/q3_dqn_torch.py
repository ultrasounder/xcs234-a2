import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger, join
from utils.test_env import EnvTest
from .q1_schedule import LinearExploration, LinearSchedule
from .q2_linear_torch import Linear

import yaml

yaml.add_constructor("!join", join)

config_file = open("config/q3_dqn.yml")
config = yaml.load(config_file, Loader=yaml.FullLoader)

############################################################
# Problem 3: Implementing DeepMind's DQN
############################################################


class NatureQN(Linear):
    """
    Implementation of DeepMind's Nature paper, please consult the methods section
    of the paper linked below for details on model configuration.
    (https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
    """

    ############################################################
    # Problem 3a: initialize_models

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The in_channels 
        to Conv2d networks will n_channels * self.config["hyper_params"]["state_history"]

        Args:
            q_network (torch model): variable to store our q network implementation

            target_network (torch model): variable to store our target network implementation

        TODO:
             (1) Set self.q_network to the architecture defined in the Nature paper associated to this question.
                Padding isn't addressed in the paper but here we will apply padding of size 2 to each dimension of
                the input to the first conv layer (this should be an argument in nn.Conv2d).
            (2) Set self.target_network to be the same configuration self.q_network but initialized from scratch
            (3) Be sure to use nn.Sequential in your implementation.

        Hints:
            (1) Start by figuring out what the input size is to the networks.
            (2) Simply setting self.target_network = self.q_network is incorrect.
            (3) The following functions might be useful
                - nn.Sequential (https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
                - nn.Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
                - nn.ReLU (https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
                - nn.Flatten (https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)
                - nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        ### START CODE HERE ###
        # Calculate the input channels(original channels * history length)
        in_channels = n_channels * config["hyper_params"]["state_history"]
        # Create Q Network
        self.q_network = nn.Sequential(
            # First conv layer with padding=2 as specified in the assignment
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Third conv layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            #Flatten the output from convolutional layers
            nn.Flatten(),
            # Fully connected layer
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        # Create Target Network
        self.target_network = nn.Sequential(   
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        # Initialize target network weights
        ### END CODE HERE ###

    ############################################################
    # Problem 3b: get_q_values

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state (torch tensor): shape = (batch_size, img height, img width,
                                            nchannels x config["hyper_params"]["state_history"])

            network (str): The name of the network, either "q_network" or "target_network"

        Returns:
            out (torch tensor): shape = (batch_size, num_actions)

        TODO:
            Perform a forward pass of the input state through the selected network
            and return the output values.


        Hints:
            (1) You can forward a tensor through a network by simply calling it (i.e. network(tensor))
            (2) Look up torch.permute (https://pytorch.org/docs/stable/generated/torch.permute.html)
        """
        out = None

        ### START CODE HERE ###
        #Select which network to use
        if network == "q_network":
            net = self.q_network
        elif network == "target_network":
            net = self.target_network
        else:
            raise ValueError("Not a valid network name")
        #Pytorch expects input in format (batch_size, channels, height, width)
        #But our state is in format (batch_size, height, width, channels)
        #So we need to permute the dimensions
        state = state.permute(0, 3, 1, 2)
        #Pass the state through the network
        out = net(state)
        ### END CODE HERE ###

        return out
