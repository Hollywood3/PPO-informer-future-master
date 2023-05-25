from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from model.model import Multihead_LSTM_informer_DataEmbedding, CFG
import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int = 128,
        last_layer_dim_pi: int = 1,
        last_layer_dim_vf: int = 1,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # self.Actor = Multihead_LSTM_informer_DataEmbedding(CFG)
        # self.Critic = Multihead_LSTM_informer_DataEmbedding(CFG)

        # Policy network
        self.policy_net = nn.Sequential(
            # nn.Linear(feature_dim, 64), nn.ReLU(),
            nn.Linear(128, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            # nn.Linear(feature_dim, 64), nn.ReLU(),
            nn.Linear(128, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # features = self.Actor(features[:, :, :-4], features[:, :, -4:])
        # features = features.view(-1, 832)
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        # features = self.Critic(features[:, :, :-4], features[:, :, -4:])
        # features = features.view(-1, 832)
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

# from stable_baselines3.common.env_util import make_vec_env
# import time
# env = make_vec_env("CartPole-v1", n_envs=4)
# model = PPO_model(CustomActorCriticPolicy, env, verbose=1)
# start_time = time.time()
# model.learn(300000)
# end_time = time.time()
# print(end_time - start_time)


import gym
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomRNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        self.Actor = Multihead_LSTM_informer_DataEmbedding(CFG)
        self.fl_ = nn.Flatten(),


        self.linear = nn.Sequential(nn.Linear(13440, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.permute(0, 2, 1)
#         print(observations.shape, observations[:, :, :-5].shape, observations[:, :, -5:].shape)
        observations = self.Actor(observations[:, :, :-5], observations[:, :, -5:])
        observations = observations.view(-1, 13440)
        return self.linear(observations)

# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=64),
# )
# model = PPO_model("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
# model.learn(1000)

def policy_kwargs():
    policy_kwargs = dict(
        features_extractor_class=CustomRNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    return policy_kwargs