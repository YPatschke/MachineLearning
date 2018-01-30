import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats.stats import pearsonr
from scipy.stats import kurtosis, skew

from cesium import datasets
from cesium import featurize

import cesium as cs

# features to extract with Cesium
features_to_extract_with_cesium = ["mean",
                   "amplitude",
                   "max_slope",
                   "maximum",
                   "median",
                   "minimum",
                   "skew",
                   "std"]

# number of timestep of the model and users data to process
users=150

# recover files of stored predicted and true trajectories from a model
filename = 'C:/Users/Yannick/Desktop/finalTest/Linear/T1/LinearPlotTrue.npy'
filename2 = 'C:/Users/Yannick/Desktop/finalTest/Linear/T1/LinearPlotPrediction.npy'
plotTrue=np.load(filename)
plotPrediction=np.load(filename2)


# lists to store all the metrics per user from true trajectories
t_mean = []
t_median = []
t_max = []
t_min = []
t_unique = []
t_skew = []
t_amplitude = []
t_max_slope = []
t_std = []
# lists to store all the metrics per user from predicted trajectories
p_mean = []
p_median = []
p_max = []
p_min = []
p_unique = []
p_skew = []
p_amplitude = []
p_max_slope = []
p_std = []
# 'corrs' will store all the Correlation an P-Value between true and predicted trajectories
corrs = []
# 'no_std' will just count the number of predicted trajectories which do not have variance
no_std = 0

# loop over users to compute metrics and append them
for user in range(users):
    # recover the data from the trajectories of the current user (true and predicted)
    t=np.array(plotTrue[user])
    p=np.array(plotPrediction[user])

    # extract fetures from the true trajectory
    timeT = [i+1 for i in range(len(t))]
    fset_cesium = featurize.featurize_time_series(times=timeT,
                                                  values=t,
                                                  errors=None,
                                                  features_to_use=features_to_extract_with_cesium)
    # append the metrics to their related list
    t_mean.append(fset_cesium["mean"][0][0])
    t_median.append(fset_cesium["median"][0][0])
    t_max.append(fset_cesium["maximum"][0][0])
    t_min.append(fset_cesium["minimum"][0][0])
    t_unique.append(len(np.unique(t)))
    t_skew.append(fset_cesium["skew"][0][0])
    t_amplitude.append(fset_cesium["amplitude"][0][0])
    t_max_slope.append(fset_cesium["max_slope"][0][0])
    t_std.append(fset_cesium["std"][0][0])

    # extract fetures from the predicted trajectory
    timeP = [i+1 for i in range(len(p))]
    fset_cesium = featurize.featurize_time_series(times=timeP,
                                                  values=p,
                                                  errors=None,
                                                  features_to_use=features_to_extract_with_cesium)
    # append the metrics to their related list
    p_mean.append(fset_cesium["mean"][0][0])
    p_median.append(fset_cesium["median"][0][0])
    p_max.append(fset_cesium["maximum"][0][0])
    p_min.append(fset_cesium["minimum"][0][0])
    p_unique.append(len(np.unique(p)))
    p_skew.append(fset_cesium["skew"][0][0])
    p_amplitude.append(fset_cesium["amplitude"][0][0])
    p_max_slope.append(fset_cesium["max_slope"][0][0])
    p_std.append(fset_cesium["std"][0][0])

    # compute Correlation and P-value between true and predicted trajectories with pearsonr
    corr=pearsonr(t,p)
    if not math.isnan(corr[0]):
        corrs.append(pearsonr(t,p))
    else:
        no_std += 1

# transfrom all the variable to numpy array
t_mean = np.array(t_mean)
t_median = np.array(t_median)
t_max = np.array(t_max)
t_min = np.array(t_min)
t_unique = np.array(t_unique)
t_skew = np.array(t_skew)
t_amplitude = np.array(t_amplitude)
t_max_slope = np.array(t_max_slope)
t_std = np.array(t_std)

p_mean = np.array(p_mean)
p_median = np.array(p_median)
p_max = np.array(p_max)
p_min = np.array(p_min)
p_unique = np.array(p_unique)
p_skew = np.array(p_skew)
p_amplitude = np.array(p_amplitude)
p_max_slope = np.array(p_max_slope)
p_std = np.array(p_std)

corrs = np.array(corrs)

# show the average of each metrics
print("="*100)
print("Mean:", t_mean.mean(), p_mean.mean())
print("Median:", t_median.mean(), p_median.mean())
print("Max:", t_max.mean(), p_max.mean())
print("Min:", t_min.mean(), p_min.mean())
print("Unique:", t_unique.mean(), p_unique.mean())
print("Std:", t_std.mean(), p_std.mean())
print("Skew:", t_skew.mean(), p_skew.mean())
print("Amplitude:", t_amplitude.mean(), p_amplitude.mean())
print("Max_slope:", t_max_slope.mean(), p_max_slope.mean())
print("Corr:", np.mean(corrs, axis=0))
print("No std:", no_std)
print("="*100)

#final table to store measures
table = [[0 for i in range(2)] for j in range(11)] # matrix 11x2 (11 measures for prediction and true trajectories)

# compute the average of each metrics and store them in the final table variable
table[0][0]=p_mean.mean()
table[1][0]=p_median.mean()
table[2][0]=p_max.mean()
table[3][0]=p_min.mean()
table[4][0]=p_unique.mean()
table[5][0]=p_std.mean()
table[6][0]=p_skew.mean()
table[7][0]=p_amplitude.mean()
table[8][0]=p_max_slope.mean()
if str(np.mean(corrs, axis=0)) == str(1e400*0): # to avoid the problem of NaN value
    table[9][0]=-1
    table[10][0]=-1
else:
    table[9][0]=np.mean(corrs, axis=0)[0]
    table[10][0]=np.mean(corrs, axis=0)[1]

table[0][1]=t_mean.mean()
table[1][1]=t_median.mean()
table[2][1]=t_max.mean()
table[3][1]=t_min.mean()
table[4][1]=t_unique.mean()
table[5][1]=t_std.mean()
table[6][1]=t_skew.mean()
table[7][1]=t_amplitude.mean()
table[8][1]=t_max_slope.mean()
table[9][1]=pearsonr(t,t)[0]
table[10][1]=pearsonr(t,t)[1]

# transform variable, arround values and save it
table = np.array(table)
table = np.around(table, decimals=3)
np.save("measuresTable.npy", table)