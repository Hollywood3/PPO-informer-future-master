from PPO_model.PPO import PPO_
from env.FutureEnv import FutureTradingEnv
from stable_baselines3.common.logger import configure
import pandas as pd
from util import add_features
import os

model = PPO.load("ppo_cartpole")
env3 = StockTradingEnv(df=data)
obs = env3.reset()
for i in range(1000):
    action = model.predict(obs)
    obs, rewards, dones, info = env3.step(action)
    env3.render()
    if dones : break
