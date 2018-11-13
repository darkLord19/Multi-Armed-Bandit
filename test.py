# Importing local functions
from random_selection import random_selection
from upper_confidence_bound import upper_confidence_bound
from thompson_sampling import thompson_sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Plot histogram of selected ads for each method
def plot(data, st, idx):
    plt.figure(idx)
    arr = plt.hist(data, ec='black')
    plt.title(st)
    plt.xlabel('Ad Number')
    plt.ylabel('Shown for # times')
    for i in range(10):
        plt.text(arr[1][i],arr[0][i]+(idx*10),str(int(arr[0][i])))

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

rs_total_reward, rs_ads_selected = random_selection(dataset)
ucb_total_reward, ucb_ads_selected = upper_confidence_bound(dataset)
th_total_reward, th_ads_selected = thompson_sampling(dataset)

rst = 'Random Selection Algorithm\nAds CTR: ' + str(rs_total_reward)
ust = 'Upper Bound Confidence Algorithm\nAds CTR: ' + str(ucb_total_reward)
tst = 'Thompson Sampling Algorithm\nAds CTR: ' + str(th_total_reward)

plot(rs_ads_selected, rst, 1)
plot(ucb_ads_selected, ust, 2)
plot(th_ads_selected, tst, 3)

plt.show()