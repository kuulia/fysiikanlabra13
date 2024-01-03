import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler

discarded_channels = 25 #the first n channels to discard 

#helper function for parsing data from csv
def read_data(filename: str) -> pd.DataFrame:
    data = pd.read_csv(filename, skiprows=8+discarded_channels) #header + first peak
    data.columns = [f'Channels {discarded_channels+1}-256']
    data.index = range(discarded_channels+1, 257)
    data.index.name = 'Channels'
    return data

#background taken before all measurements
bg_5min = read_data('background.csv')
bg_5min = bg_5min*1.086956 # 276,269s measurement -> 300s

#background taken after all measurements
bg_10min = read_data('background10min.csv') #600s measurement
bg_10min = 0.5 *bg_10min # normalize to 300s 

bg_5min_avg = (bg_5min + bg_10min) / 2 
co60 = read_data('co60.csv')
co60 = co60 / 1.01190 # 304s -> 300s
cs137 = read_data('cs137.csv') # 309s measurement
cs137 = cs137 / 1.031416 # 309s -> 300s
k40 = read_data('k40.csv') # 597s 
k40 = 0.502513 * k40
lyijy_cs137 = read_data('lyijy_cs137.csv') #300s measurement
na22 = read_data('na22.csv') #603
na22 = 0.498061 * na22 

data_names = ['co60_nn', 'cs137_nn', 'lyijy_cs137_nn', 'na22_nn', 'k40_nn']
#_nn = noise normalized
data_5min = [co60, cs137, lyijy_cs137, na22, k40]

#normalize for noise
data_noise_normalized = []
for i, data in enumerate(data_5min):
    data_noise_normalized.append(data - bg_5min)

for i, data in enumerate(data_noise_normalized):
    #save nn data to folder
    data.to_csv(f'nn_data/{data_names[i]}.csv')

    #bar plot
    plt.bar(list(data.index), data[f'Channels {discarded_channels+1}-256'])
    plt.xlabel('Channel (idx)')
    plt.ylabel('Counts (N)')
    plt.savefig(f'plots/{data_names[i]}_barplot.png')
    plt.close()

plt.bar(list(bg_5min_avg.index), bg_5min_avg[f'Channels {discarded_channels+1}-256'])
plt.savefig('plots/5min_bg_barplot.png')
plt.close()