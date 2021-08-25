import pandas as pd
import psycopg2 as pg
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grd
from  matplotlib.ticker import PercentFormatter
from matplotlib import ticker
from textwrap import wrap
import time
import random
import math
import feather
from copy import deepcopy

from scipy.stats import binned_statistic
from scipy import stats
from sklearn.metrics import confusion_matrix

import seaborn as sns
import pycountry_convert as pc

import torch.nn as nn
import torch.nn.functional as F
import torch

from pandas.api.types import CategoricalDtype

import os


sns.set_theme()


import warnings
warnings.filterwarnings('ignore')


import sys; sys.path.append("/dhi_work/wum/warfarin_rl")
from utils.helper_functions import *



def create_tuples(buffer_data, state_dim, state_method, num_classes):
    
    num_pats = buffer_data['USUBJID_O_NEW'].nunique()
    max_seq = buffer_data.groupby('USUBJID_O_NEW').count().max()[0]
    print(f"The longest trajectory is: {max_seq}")
    
    ##############################
    # Demog 
    ##############################
    feat_dim = state_dim
    tensor = torch.ones(())
    dem = tensor.new_empty((num_pats, max_seq, feat_dim))

    ##############################
    # Obs
    ##############################
    feat_dim = 1
    tensor = torch.ones(())
    obs = tensor.new_empty((num_pats, max_seq, feat_dim))
    
    ##############################
    # Rewards
    ##############################
    feat_dim = 1
    tensor = torch.ones(())
    rewards = tensor.new_empty((num_pats, max_seq, feat_dim))

    ##############################
    # Actions
    ##############################
    feat_dim = 1
    tensor = torch.ones(())
    actions = tensor.new_empty((num_pats, max_seq, feat_dim))
 
    ##############################
    # Length
    ##############################
    feat_dim = 1
    tensor = torch.ones(())
    lengths = tensor.new_empty((num_pats, max_seq, feat_dim))

    ##############################
    # k
    ##############################
    feat_dim = 1
    tensor = torch.ones(())
    k = tensor.new_empty((num_pats, max_seq, feat_dim))

    ##############################
    # Done flag
    ##############################
    feat_dim = 1
    tensor = torch.ones(())
    done = tensor.new_empty((num_pats, max_seq, feat_dim))

    
    buffer_data["EVENT_NEXT_STEP"] = np.minimum(1, buffer_data[Constants.neg_reward_events].shift(-1).sum(axis=1))
    
    i = 0
    for pat_id in buffer_data['USUBJID_O_NEW'].unique():

        subset = buffer_data[buffer_data['USUBJID_O_NEW'] == pat_id]
        temp_state = ReplayBuffer.get_state(subset, state_method, verbose=False)
        temp_state_tensor = torch.FloatTensor(temp_state)

        sample_dem = temp_state_tensor[:, 1:]
        sample_obs = temp_state_tensor[:, 0].resize(sample_dem.size()[0], 1)
        sample_actions = torch.FloatTensor(subset["ACTION"].values).resize(sample_dem.size()[0], 1)
        sample_rewards = torch.FloatTensor(subset["REWARD"].values).resize(sample_dem.size()[0], 1)
        sample_k = torch.FloatTensor(subset["k"].values).resize(sample_dem.size()[0], 1)
        sample_done = torch.FloatTensor(subset["DONE"].values).resize(sample_dem.size()[0], 1)
        
        traj_length = sample_obs.size()[0]        
        pad_size = max_seq - sample_obs.size()[0]
        sample_obs = F.pad(input=sample_obs, pad=(0,0,0,pad_size), value=0, mode="constant")
        sample_dem = F.pad(input=sample_dem, pad=(0,0,0,pad_size), value=0, mode="constant")
        sample_actions = F.pad(input=sample_actions, pad=(0,0,0,pad_size), value=0, mode="constant")
        sample_rewards = F.pad(input=sample_rewards, pad=(0,0,0,pad_size), value=0, mode="constant")
        sample_k = F.pad(input=sample_k, pad=(0,0,0,pad_size), value=0, mode="constant")
        sample_done = F.pad(input=sample_done, pad=(0,0,0,pad_size), value=0, mode="constant")
        
        obs[i] = sample_obs
        dem[i] = sample_dem
        actions[i] = sample_actions
        rewards[i] = sample_rewards
        lengths[i] = traj_length
        k[i] = sample_k
        done[i] = sample_done 
        
        i += 1

        if i % 500 == 0:
            print(f"Passed {i} patients")
        
        if i >= 10000:
            print("Force breaking at 10000 patients")
            break
            
    labels = actions

    one_hot_target = (labels == torch.arange(num_classes).reshape(1, num_classes)).float()
    actions = one_hot_target

    obs = obs[:, :, 0].resize(obs.size()[0],obs.size()[1], 1)

    return obs, dem, actions, rewards, lengths, k, done



Constants.neg_reward_events = ["STROKE", "HEM_STROKE", "MAJOR_BLEED"]
state_method = 19
state_dim = 52
num_actions = 7
incl_events = False

buffer_suffix = "smdp_interpolated_full"

buffer_name = f"buffer_data_{buffer_suffix}"
suffix = f"actions_{num_actions}_state_{state_dim}_{buffer_suffix}"
# train_buffer, val_buffer, test_buffer, events_buffer = SMDPReplayBuffer.load_buffers(buffer_name, suffix)


# In[16]:


root_dir = "../"
is_ais = False


# In[ ]:

print(f"Loading buffers")
train_buffer = SMDPReplayBuffer(filename=buffer_name + "_train", root_dir=root_dir)
val_buffer = SMDPReplayBuffer(filename=buffer_name + "_valid", root_dir=root_dir)
test_buffer = SMDPReplayBuffer(filename=buffer_name + "_test", root_dir=root_dir)

train_buffer.load_buffer(buffer_name=suffix, dataset="train", ais=is_ais)
val_buffer.load_buffer(buffer_name=suffix, dataset="valid", ais=is_ais)
test_buffer.load_buffer(buffer_name=suffix, dataset="test", ais=is_ais)

train_buffer.data = feather.read_dataframe(train_buffer.data_path)
cumcount_id = train_buffer.data.groupby("USUBJID_O_NEW").cumcount()
train_buffer.data["DONE"] = (cumcount_id > cumcount_id.shift(-1))
train_buffer.data["DONE"] = np.logical_or(train_buffer.data["DONE"].shift(-1), train_buffer.data["DONE"]).fillna(False)
train_buffer.data["DONE"].sum()
train_buffer.data["DONE"] = (~train_buffer.data["DONE"]).astype(int)

test_buffer.data = feather.read_dataframe(test_buffer.data_path)
cumcount_id = test_buffer.data.groupby("USUBJID_O_NEW").cumcount()
test_buffer.data["DONE"] = (cumcount_id > cumcount_id.shift(-1))
test_buffer.data["DONE"] = np.logical_or(test_buffer.data["DONE"].shift(-1), test_buffer.data["DONE"]).fillna(False)
test_buffer.data["DONE"].sum()
test_buffer.data["DONE"] = (~test_buffer.data["DONE"]).astype(int)

val_buffer.data = feather.read_dataframe(val_buffer.data_path)
cumcount_id = val_buffer.data.groupby("USUBJID_O_NEW").cumcount()
val_buffer.data["DONE"] = (cumcount_id > cumcount_id.shift(-1))
val_buffer.data["DONE"] = np.logical_or(val_buffer.data["DONE"].shift(-1), val_buffer.data["DONE"]).fillna(False)
val_buffer.data["DONE"].sum()
val_buffer.data["DONE"] = (~val_buffer.data["DONE"]).astype(int)


print(f"Test: {test_buffer.data.shape}, Valid: {val_buffer.data.shape}, Train: {train_buffer.data.shape}")

buffer_suffix = "smdp_interpolated_full"

# print(f"Creating validation tuples")
# val_buffer.data = val_buffer.data.dropna(subset=["INR_VALUE", "WEIGHT"])
# buffer_data = deepcopy(val_buffer.data)
# obs, dem, actions, rewards, lengths, k, done = create_tuples(buffer_data, state_dim=state_dim-1, state_method=state_method, num_classes=num_actions)
# torch.save((obs, dem, actions, rewards, lengths, k, done), f"../data/clean_data/rl4h_tuples/tuples_bcq/{buffer_suffix}/valid_ais_tuples")

# print(f"Creating test tuples")
# test_buffer.data = test_buffer.data.dropna(subset=["INR_VALUE", "WEIGHT"])
# buffer_data = deepcopy(test_buffer.data)
# obs, dem, actions, rewards, lengths, k, done = create_tuples(buffer_data, state_dim=state_dim-1, state_method=state_method, num_classes=num_actions)
# torch.save((obs, dem, actions, rewards, lengths, k, done), f"../data/clean_data/rl4h_tuples/tuples_bcq/{buffer_suffix}/test_ais_tuples")

print(f"Creating train tuples")
train_buffer.data = train_buffer.data.dropna(subset=["INR_VALUE", "WEIGHT"])
buffer_data = deepcopy(train_buffer.data)
obs, dem, actions, rewards, lengths, k, done = create_tuples(buffer_data, state_dim=state_dim-1, state_method=state_method, num_classes=num_actions)
torch.save((obs, dem, actions, rewards, lengths, k, done), f"../data/clean_data/rl4h_tuples/tuples_bcq/{buffer_suffix}/train_ais_tuples_force_10000")
