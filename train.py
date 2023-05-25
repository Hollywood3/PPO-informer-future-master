# from env.FutureEnv import FutureTradingEnv
from PPO_model.PPO import PPO_
from stable_baselines3.common.logger import configure
import pandas as pd
from util import add_features
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == '__main__':
    data = pd.read_csv('E:\金融数据\甲醛.csv')
    data = add_features(data)
    data = data.sort_values('date')
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month / 12 - 0.5
    data['day'] = data['date'].dt.day / 30 - 0.5
    data['weekday'] = data['date'].dt.weekday / 6 - 0.5
    data['hour'] = data['date'].dt.hour / 23 - 0.5
    data['minute'] = data['date'].dt.minute / 59 - 0.5

    model = PPO_(data)
    tmp_path = "/tmp/sb3_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=150000)
    model.save("ppo_cartpole")
