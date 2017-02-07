import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')

dataset = {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}
new_features = [5,7]

def k_nearest_neighbors(data, predict, k=3):
    # by default k =5 in sci-kit
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    #print (Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

df = pd.read_csv("/Users/kachunfung/python/ml/breast-cancer-wisconsin.data.txt")
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist() # if reuse = float, avoid string
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data: # i[-1] = label
    # {2:[feature], 4:[]feature}
    train_set[i[-1]].append(i[:-1]) # i[:-1] = feature set

for i in test_data: # i[-1] = label
    # {2:[feature], 4:[]feature}
    test_set[i[-1]].append(i[:-1]) # i[:-1] = feature set

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]: # data = list of feature
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct +=1
        total +=1

print('Accuracy:', float(correct)/total)
