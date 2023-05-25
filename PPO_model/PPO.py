from env.FutureEnv import FutureTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from PPO_model.CustomActorCriticPolicy import policy_kwargs, CustomActorCriticPolicy


def PPO_(data):

    env = make_vec_env(lambda: FutureTradingEnv(data), n_envs=4)
    policy_kwarg = policy_kwargs()
    model = PPO(CustomActorCriticPolicy, env, verbose=0, policy_kwargs=policy_kwarg, batch_size=512)

    return model
