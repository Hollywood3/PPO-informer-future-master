import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import math

def add_features(feats):
    feats["return_15minute"] = feats["close"].pct_change(15)
    feats["return_30minute"] = feats["close"].pct_change(30)
    feats["return_60minute"] = feats["close"].pct_change(60)
    feats["return_120minute"] = feats["close"].pct_change(120)
    feats["volatility_15minute"] = (
        np.log(feats["close"]).diff().rolling(15).std()
    )
    feats["volatility_30minute"] = (
        np.log(feats["close"]).diff().rolling(30).std()
    )
    feats["volatility_60minute"] = (
        np.log(feats["close"]).diff().rolling(60).std()
    )
    feats["volatility_120minute"] = (
        np.log(feats["close"]).diff().rolling(120).std()
    )
    feats["MA_gap_15minute"] = feats["close"] / (
        feats["close"].rolling(15).mean()
    )
    feats["MA_gap_30minute"] = feats["close"] / (
        feats["close"].rolling(30).mean()
    )
    feats["MA_gap_60minute"] = feats["close"] / (
        feats["close"].rolling(60).mean()
    )
    feats["MA_gap_120minute"] = feats["close"] / (
        feats["close"].rolling(120).mean()
    )

    return feats