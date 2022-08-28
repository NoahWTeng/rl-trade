from time import time
from utils import (
    crypto_download_data,
    load_dataset,
    prepare_data,
    data_split,
    add_all_indicator,
    split_data,
    low_variance,
    normalizing,
)
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from datetime import timezone, datetime
from train import test, train
from envs import CryptoEnv


path = "dataset"
exchange = "Bitfinex"
coin = "BTC"
fiat = "USD"
timeframe = "1h"
filename = f"{path}/{exchange}_{coin}_{fiat}_{timeframe}.csv"


# crypto_download_data(path, exchange, coin, fiat, timeframe, True)


dataset = load_dataset(file_name=filename)
dataset = prepare_data(dataset)


"""
ADD INDICATORS
"""

dataset = add_all_indicator(dataset)

############################################################
# Remove features with low variance before splitting the dataset
############################################################

dataset = low_variance(dataset)


############################################################
# Split dataset
############################################################

X_train, X_test, X_valid, y_train, y_test, y_valid = split_data(dataset)

"""
TRAIN DATESET
"""
train_dataset = dataset.copy()
train_dataset_normalized = normalizing(train_dataset)

# print(train_dataset)
# print(train_dataset_normalized)


"""
TESTING DATESET
"""
testing_start = datetime(2019, 10, 19)
testing_end = datetime(2021, 10, 19)
testing_start_timestamp = testing_start.replace(tzinfo=timezone.utc).timestamp()
testing_end_timestamp = testing_end.replace(tzinfo=timezone.utc).timestamp()
testing_dataset = data_split(
    dataset.copy(), int(testing_start_timestamp), int(testing_end_timestamp)
)
testing_dataset_normalized = normalizing(testing_dataset)
# print(testing_dataset)
# print(testing_dataset_normalized)


"""
EVALUATION DATASET
"""
evaluation_start = datetime(2018, 5, 19)
evaluation_end = datetime(2019, 10, 19)
evaluation_start_timestamp = evaluation_start.replace(tzinfo=timezone.utc).timestamp()
evaluation_end_timestamp = evaluation_end.replace(tzinfo=timezone.utc).timestamp()
evaluation_dataset = data_split(
    dataset.copy(), int(evaluation_start_timestamp), int(evaluation_end_timestamp)
)

evaluation_dataset_normalized = normalizing(evaluation_dataset)
# print(evaluation_dataset)
# print(evaluation_dataset_normalized)


TICKER_LIST = ["BTCUSD"]
env = CryptoEnv
TRAIN_START_DATE = "2021-09-01"
TRAIN_END_DATE = "2021-10-02"

TEST_START_DATE = "2021-09-21"
TEST_END_DATE = "2021-09-30"

INDICATORS = [
    "macd",
    "rsi",
    "cci",
    "dx",
]  # self-defined technical indicator list is NOT supported yet

ERL_PARAMS = {
    "learning_rate": 2**-15,
    "batch_size": 2**11,
    "gamma": 0.99,
    "seed": 312,
    "net_dimension": 2**9,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 1,
}

train(
    dataset=train_dataset,
    dataset_normalized=train_dataset_normalized,
    time_interval="1h",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    current_working_dir="./test_ppo",
    erl_params=ERL_PARAMS,
    break_step=5e4,
    if_vix=False,
)
