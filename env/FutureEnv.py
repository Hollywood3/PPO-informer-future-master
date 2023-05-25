import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import math

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
MAX_DAY_CHANGE = 1

INITIAL_ACCOUNT_BALANCE = 10000


class FutureTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(FutureTradingEnv, self).__init__()

        self.df = df

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([-1]), high=np.array([1]), dtype=np.float32)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(28, 120), dtype=np.float32)

        self.open_time_index = list(self.df[(self.df['date'].dt.hour == 21) & (self.df['date'].dt.minute == 1)].index)
        self.open_time_index_len = range(len(self.open_time_index) - 1)
        self._std_mean_()

    def _std_mean_(self):

        index_ = list(self.df.columns)
        self.std_dir = {}
        self.mean_dir = {}
        for i in index_:
            if i != 'date':
                self.mean_dir[f'{i}_mean'] = self.df[i].mean()
                self.std_dir[f'{i}_std'] = self.df[i].std()

    def _next_observation(self):
        self.df.loc[self.current_step + self.start_time, 'balance'] = self.balance / 10000
        self.df.loc[self.current_step + self.start_time, 'shares_held'] = self.shares_held
        self.df.loc[self.current_step + self.start_time, 'cost_basis'] = self.cost_basis / 3000
        self.df.loc[self.current_step + self.start_time, 'total_shares_sold'] = self.total_shares_sold / 100
        obs = np.array([
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'open'] -
             self.mean_dir['open_mean']) / self.std_dir['open_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'high'] -
             self.mean_dir['high_mean']) / self.std_dir['high_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'low'] -
             self.mean_dir['low_mean']) / self.std_dir['low_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'close'] -
             self.mean_dir['close_mean']) / self.std_dir['close_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'volume'] -
             self.mean_dir['volume_mean']) / self.std_dir['volume_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'money'] -
             self.mean_dir['money_mean']) / self.std_dir['money_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'open_interest'] - self.mean_dir['open_interest_mean']) / self.std_dir['open_interest_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'return_15minute'] - self.mean_dir['return_15minute_mean']) / self.std_dir['return_15minute_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'return_30minute'] - self.mean_dir['return_30minute_mean']) / self.std_dir['return_30minute_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'return_60minute'] - self.mean_dir['return_60minute_mean']) / self.std_dir['return_60minute_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'return_120minute'] - self.mean_dir['return_120minute_mean']) / self.std_dir['return_120minute_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'volatility_15minute'] - self.mean_dir['volatility_15minute_mean']) / self.std_dir[
                'volatility_15minute_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'volatility_30minute'] - self.mean_dir['volatility_30minute_mean']) / self.std_dir[
                'volatility_30minute_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'volatility_60minute'] - self.mean_dir['volatility_60minute_mean']) / self.std_dir[
                'volatility_60minute_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'volatility_120minute'] - self.mean_dir['volatility_120minute_mean']) / self.std_dir[
                'volatility_120minute_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'MA_gap_15minute'] - self.mean_dir['MA_gap_15minute_mean']) / self.std_dir['MA_gap_15minute_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'MA_gap_30minute'] - self.mean_dir['MA_gap_30minute_mean']) / self.std_dir['MA_gap_30minute_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'MA_gap_60minute'] - self.mean_dir['MA_gap_60minute_mean']) / self.std_dir['MA_gap_60minute_std'],
            (self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
             'MA_gap_120minute'] - self.mean_dir['MA_gap_120minute_mean']) / self.std_dir['MA_gap_120minute_std'],
            self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'balance'],
            self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'shares_held'],
            self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'cost_basis'],
            self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time,
            'total_shares_sold'],
            self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'month'],
            self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'day'],
            self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'weekday'],
            self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'hour'],
            self.df.loc[self.current_step + self.start_time - 119:self.current_step + self.start_time, 'minute'],
        ])
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step + self.start_time + 1, "close"]
        # current_price = random.uniform(
        #     self.df.loc[self.current_step + self.start_time+1, "open"], self.df.loc[self.current_step + self.start_time+1, "close"])
        self.now_open = math.floor(current_price)
        current_price = self.now_open
        action_type = action[0]

        if 0.5 < action_type:

            if self.shares_held == 0:
                if self.balance > current_price * 10 * 0.12 + 2:
                    self.shares_held = 1
                    self.balance -= current_price * 10 * 0.12 + 0
                    self.cost_basis = current_price * 10 * 0.12
                    self.price_basis = current_price
                    self.total_shares_sold = self.total_shares_sold
            elif self.shares_held == 1:
                self.shares_held = 1
                self.balance = self.balance
                # self.cost_basis = 0
                self.total_shares_sold = self.total_shares_sold
            elif self.shares_held == -1:
                self.shares_held = 0
                self.balance += self.cost_basis + (current_price - self.price_basis) * 10 - 8
                self.cost_basis = 0
                self.price_basis = 0
                self.total_shares_sold += 1
                self.df['balance'] = self.balance
                self.df['shares_held'] = 0
                self.df['cost_basis'] = 0

        elif action_type < -0.5:
            if self.shares_held == 0:
                if self.balance > current_price * 10 * 0.12 + 2:
                    self.shares_held = -1
                    self.balance -= current_price * 10 * 0.12 + 0
                    self.cost_basis = current_price * 10 * 0.12
                    self.price_basis = current_price
                    self.total_shares_sold = self.total_shares_sold
            elif self.shares_held == 1:
                self.shares_held = 0
                self.balance += self.cost_basis + (self.price_basis - current_price) * 10 - 8
                self.cost_basis = 0
                self.price_basis = 0
                self.total_shares_sold += 1
                self.df['balance'] = self.balance
                self.df['shares_held'] = 0
                self.df['cost_basis'] = 0
            elif self.shares_held == -1:
                self.shares_held = -1
                self.balance = self.balance
                # self.cost_basis = 0
                self.total_shares_sold = self.total_shares_sold
        else:
            if self.shares_held == 0:
                self.shares_held = 0
                self.balance = self.balance
                self.cost_basis = 0
                self.price_basis = 0
                self.total_shares_sold = self.total_shares_sold
            elif self.shares_held == 1:
                #                 self.shares_held = 0
                #                 self.balance += current_price * 10 * 0.12 - 0
                #                 self.cost_basis = 0
                #                 self.total_shares_sold += 1
                self.shares_held = 1
                self.balance = self.balance
                self.cost_basis = self.cost_basis
                self.total_shares_sold = self.total_shares_sold
            elif self.shares_held == -1:
                #                 self.shares_held = 0
                #                 self.balance += self.cost_basis * 2 - current_price * 10 * 0.12 - 0
                #                 self.cost_basis = 0
                #                 self.total_shares_sold += 1
                self.shares_held = -1
                self.balance = self.balance
                self.cost_basis = self.cost_basis
                self.total_shares_sold = self.total_shares_sold

        if self.shares_held == 0:
            self.net_worth = self.balance
        elif self.shares_held == 1:
            self.net_worth = self.balance + self.cost_basis + (self.price_basis - current_price) * 10
        elif self.shares_held == -1:
            self.net_worth = self.balance + self.cost_basis + (current_price - self.price_basis) * 10

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        done = False

        self.current_step += 1

        if self.current_step > (self.end_time - self.start_time - 1):
            # self.current_step = 0  # loop training
            done = True

        # profits
        reward = self.net_worth - self.worth
        # reward = 1 if reward > 0 else -100

        if self.shares_held == 0:
            reward += 0.1

        self.worth = self.net_worth
        if self.net_worth <= 0:
            done = True

        obs = self._next_observation()
        if obs.shape[1] != 120:
            obs = obs[:, :120]

        #         if reward>=0:
        #             reward = reward ** 0.5
        #         else:
        #             reward = (-reward) ** 0.5
        #             reward = -reward
        return obs, reward, done, {}

    def reset(self, new_df=None):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.total_shares_sold = 0
        self.cost_basis = 0
        self.price_basis = 0

        # pass test dataset to environment
        if new_df:
            self.df = new_df
            self.open_time_index = list(
                self.df[(self.df['date'].dt.hour == 21) & (self.df['date'].dt.minute == 1)].index)
            self.open_time_index_len = range(len(self.open_time_index) - 1)

        self.df['balance'] = INITIAL_ACCOUNT_BALANCE
        self.df['shares_held'] = 0
        self.df['cost_basis'] = 0
        self.df['total_shares_sold'] = 0
        index_ = random.choice(self.open_time_index_len)
        self.start_time = self.open_time_index[index_]
        self.end_time = self.open_time_index[index_ + 1]
        if self.end_time - self.start_time != 345 or self.end_time - self.start_time != 375:
            index_ = random.choice(self.open_time_index_len)
            self.start_time = self.open_time_index[index_]
            self.end_time = self.open_time_index[index_ + 1]
        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)
        self.current_step = 0
        self.old_open = self.df.loc[self.start_time, 'open']
        #         self._std_mean_()

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-' * 30)
        print(f'第几分钟: {self.current_step}')
        print(f'银行账户: {self.balance}')
        print(f'持有仓位: {self.shares_held} (总共卖出: {self.total_shares_sold})')
        print(f'价格：{self.now_open}')
        print(f'持仓成本: {self.cost_basis}')
        print(f'总价值: {self.net_worth}')
        print(f'盈利: {profit}')
        return profit
