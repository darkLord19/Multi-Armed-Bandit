# Importing local functions
from random_selection import random_selection
from upper_confidence_bound import upper_confidence_bound
from thompson_sampling import thompson_sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

rs_total_reward, rs_ads_selected = random_selection(dataset)
ucb_total_reward, ucb_ads_selected = upper_confidence_bound(dataset)
th_total_reward, th_ads_selected = thompson_sampling(dataset)

print(rs_total_reward, ucb_total_reward, th_total_reward)