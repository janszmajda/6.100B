#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:30:38 2025

@author: johnguttag
"""

"""
Features all non-neg ints
Split must be on a single feature, xi,
Split of form n <= xi <= m.

Import only numpy and random
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_text
from collections import Counter
import sys

# set line width
plt.rcParams['lines.linewidth'] = 4
# set font size for titles
plt.rcParams['axes.titlesize'] = 12
# set font size for labels on axes
plt.rcParams['axes.labelsize'] = 12
# set size of numbers on x-axis
plt.rcParams['xtick.labelsize'] = 12
# set size of numbers on y-axis
plt.rcParams['ytick.labelsize'] = 12
# set size of ticks on x-axis
plt.rcParams['xtick.major.size'] = 7
# set size of ticks on y-axis
plt.rcParams['ytick.major.size'] = 7
# set numpoints for legend``
plt.rcParams['legend.numpoints'] = 1
# set marker size
plt.rcParams['lines.markersize'] = 6


## finishing up bad statistics

def r_squared(observed, predicted):
    error = ((predicted - observed)**2).sum()
    mean_error = error / len(observed)
    return 1 - mean_error / np.var(observed)

def get_water_data(filename):
    with open(filename, 'r') as data_file:
        y_vals = []
        x_vals = []
        for _ in range(4):
            data_file.readline() # discard incomplete lines
        for line in data_file:
            split_line = line.split(',')
            x_vals.append(split_line[0])
            y_vals.append(12*float(split_line[4])) #convert feet to inches
        return np.array(x_vals), np.array(y_vals)

def plot_water_data(days, deltas, color, label):
    plt.plot(days, deltas, color, label = label)
    plt.xlabel('Months Since August 1921', size = 16)
    plt.ylabel('Change in Sea Level (inches)', size = 16)
    plt.title('Mean High Sea Level in Boston')

def make_water_plot(with_trailing = False):
    x_vals, y_vals = get_water_data('boston_sea_levels.csv')
    deltas = y_vals - y_vals[0]
    days = range(len(x_vals))
    plot_water_data(days, deltas, 'c', label = 'monthly mean')
    
    if with_trailing: # 5-month trailing average
        trailing_avg = np.convolve(deltas, np.ones(12)/12, mode='valid')
        plt.plot(days[11:], trailing_avg, label = '12 month trailing average')
        plt.legend()
        model = np.polyfit(days[11:], trailing_avg, 1)
        print('Linear model of trailing ave. =', model)
        preds = np.polyval(model, days[11:])
        r2 = r_squared(trailing_avg, preds)
        plt.plot(days[11:], preds,
                 label = f'trailing ave. model, r2 = {r2:.2f}')
        model = np.polyfit(days, deltas, 1)
        print('Linear model of monthly data =', model)
        preds = np.polyval(model, days)
        r2 = r_squared(deltas, preds)
        plt.plot(days, preds, 'r', label = f'monthly model, r2 = {r2:.2f}')
        plt.legend(fontsize = 12)
        sys.exit()
    
    # Pick some cherries
    max_delta = 0
    for d in range(800, 1000):
        if deltas[d] > max_delta:
            max_delta = deltas[d]
            max_d = d
    plt.plot((max_d, days[-1]), (deltas[max_d], deltas[days[-1]]), 'b',
             label = 'Connect two points')
    # Construct some fits
    model = np.polyfit(days, deltas, 1)
    preds = np.polyval(model, days)
    r2 = r_squared(deltas, preds)
    plt.plot(days, preds, 'r', label = f'Linear model, r2 = {r2:.2f}')
    model = np.polyfit(days[max_d:], deltas[max_d:], 1)
    preds = np.polyval(model, days[max_d:])
    plt.plot(days[max_d:], preds, 'k', label = 'Linear model')
    plt.legend(fontsize = 14)
    
make_water_plot(True)

sys.exit()

### Cancer cluster

t_stat, p_value = stats.ttest_1samp([39, 39, 37, 44, 42],
                                    popmean = 36)
print(f'p-value = {p_value:.3f}')
## code for evaluating models

def accuracy(vals):
    tn, fp, fn, tp = vals
    return (tp+tn)/(tp+tn+fp+fn)

def precision(vals):
    tn, fp, fn, tp = vals
    if tp + fp == 0:
        return 'NaN'
    else:
        return tp/(tp + fp)

def recall(vals):
    tn, fp, fn, tp = vals
    if tp == 0:
        return 0
    else:
        return tp/(tp + fn)
    
def f1_score(vals):
    tn, fp, fn, tp = vals
    return ((2*precision(tp, fp, tn, fn)*recall(tp, fp, tn, fn))/
            (precision(tp, fp, tn, fn)+recall(tp, fp, tn, fn)))

def conf_matrix(vals, title, caption = None):
    tp, tn, fp, fn = vals
    values = [[tp, fn],
              [fp, tn]]
    labels = [['TP', 'FN'],
              ['FP', 'TN']]
    colors = [['lightgreen', 'lightsalmon'],  # row: actual positive
              ['lightsalmon', 'lightgreen']]  # row: actual negative
    fig, ax = plt.subplots()
    # Draw colored rectangles
    for i in range(2):
        for j in range(2):
            rect = patches.Rectangle((j, i), 1, 1, facecolor=colors[i][j], edgecolor='black')
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, f'{labels[i][j]}\n{values[i][j]}',
                    ha='center', va='center', fontsize=12)
    # Axis formatting
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_yticklabels(['Positive', 'Negative'])
    if caption == None:
        ax.set_xlabel('Predicted label')
    else:
        ax.set_xlabel(caption)
    ax.set_ylabel('True label')
    ax.set_title(title)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()  # Make (0,0) the top-left
    ax.set_aspect('equal')
    plt.grid(False)
    plt.tight_layout()

def entropy(data):
    labels = [label for _, label in data]
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total)
                for count in counts.values())

def info_gain(data, feature_index):
    feature_names = ['outlook', 'temperature', 'humidity', 'windy']
    base_entropy = entropy(data)
    subsets = {}
    for row in data:
        key = row[0][feature_index]
        subsets.setdefault(key, []).append(row)

    subset_entropy = sum((len(subset) / len(data)) * entropy(subset)
                         for subset in subsets.values()) 
    print(f'Entropy for {feature_names[feature_index]} = {subset_entropy:.3f}')
    return base_entropy - subset_entropy

def best_split(data):
    features = len(data[0][0])
    gains = [[i, info_gain(data, i)] for i in range(features)]
    return max(gains, key=lambda x: x[1])

def majority_class(data):
    labels = [label for _, label in data]
    return Counter(labels).most_common(1)[0][0]

def build_tree(data, depth=0):
    labels = [label for _, label in data]
    if len(set(labels)) == 1:
        return labels[0]
    if not data[0][0]:
        return majority_class(data)
    best_feature, gain = best_split(data)
    if gain == 0:
        return majority_class(data)
    tree = {'feature': best_feature, 'branches': {}}
    subsets = {}
    for row in data:
        key = row[0][best_feature]
        subsets.setdefault(key, []).append(row)
    for feature_val, subset in subsets.items():
        tree['branches'][feature_val] = build_tree(subset, depth + 1)
    return tree

def classify(tree, sample):
    while isinstance(tree, dict):
        feature = tree['feature']
        value = sample[feature]
        if value not in tree['branches']:
            return None
        tree = tree['branches'][value]
    return tree

def pretty_print_tree(tree, feature_names=None, indent=''):
    if isinstance(tree, str):
        print(indent + f"Predict: {tree}")
        return

    feature = tree['feature']
    branches = tree['branches']
    feature_name = f"Feature[{feature}]"
    if feature_names:
        feature_name = feature_names[feature]

    for branch_val, subtree in branches.items():
        print(f"{indent}if {feature_name} == {branch_val}:")
        pretty_print_tree(subtree, feature_names, indent + '    ')


# Format: [[features, label)]]
# Format: [outlook, temperature, humidity, windy]
david_history = [
    [['Sunny', 'Hot', 'High', 'False'], 'No'],
    [['Sunny', 'Hot', 'High', 'True'], 'No'],
    [['Overcast', 'Hot', 'High', 'False'], 'Yes'],
    [['Rainy', 'Mild', 'High', 'False'], 'Yes'],
    [['Rainy', 'Cool', 'Normal', 'False'], 'Yes'],
    [['Rainy', 'Cool', 'Normal', 'True'], 'No'],
    [['Overcast', 'Cool', 'Normal', 'True'], 'Yes'],
    [['Sunny', 'Mild', 'High', 'False'], 'No'],
    [['Sunny', 'Cool', 'Normal', 'False'], 'Yes'],
    [['Rainy', 'Mild', 'Normal', 'False'], 'Yes'],
    [['Sunny', 'Mild', 'Normal', 'True'], 'Yes'],
    [['Overcast', 'Mild', 'High', 'True'], 'Yes'],
    [['Overcast', 'Hot', 'Normal', 'False'], 'Yes'],
    [['Rainy', 'Mild', 'High', 'True'], 'No'],
]

john_history = [
    [['Sunny', 'Hot', 'High', 'False'], 'No'],
    [['Sunny', 'Hot', 'High', 'True'], 'No'],
    [['Overcast', 'Hot', 'High', 'False'], 'Yes'],
    [['Rainy', 'Mild', 'High', 'False'], 'Yes'],
    [['Rainy', 'Cool', 'Normal', 'False'], 'No'],
    [['Rainy', 'Cool', 'Normal', 'True'], 'No'],
    [['Overcast', 'Cool', 'Normal', 'True'], 'No'],
    [['Sunny', 'Mild', 'High', 'False'], 'No'],
    [['Sunny', 'Cool', 'Normal', 'False'], 'No'],
    [['Rainy', 'Mild', 'Normal', 'False'], 'Yes'],
    [['Sunny', 'Mild', 'Normal', 'True'], 'Yes'],
    [['Overcast', 'Mild', 'High', 'True'], 'Yes'],
    [['Overcast', 'Hot', 'Normal', 'False'], 'Yes'],
    [['Rainy', 'Mild', 'High', 'True'], 'No'],
    ]

# feature_names = ['outlook', 'temperature', 'humidity', 'windy']
# tree_david = build_tree(david_history)
# print('The decision tree learned for David:')
# pretty_print_tree(tree_david, feature_names, '  ')

# tree_john = build_tree(john_history)
# print('\nThe decision tree learned from John:')
# pretty_print_tree(tree_john, feature_names, '  ')

# sample = ['Rainy', 'Cool', 'High', 'False']
# print("Prediction for sample using David tree:", classify(tree_david, sample))
# print("Prediction for sample using John tree:", classify(tree_john, sample))

# sys.exit()

## Cardiac Example

def set_seeds(val):
    random.seed(val)
    np.random.seed(val)
       
def test_model(forest, X, Y, verbose = False):
    y_hats = forest.predict(X)
    vals = confusion_matrix(Y, y_hats).ravel()
    return vals

def test_rule(df, rule):
    subset = rule(df)
    tp = subset[(subset['outcome'] == 1)].shape[0]
    fp = subset[(subset['outcome'] == 0)].shape[0]
    complement = df.loc[~df.index.isin(subset.index)]
    fn = complement[(complement['outcome'] == 1)].shape[0]
    tn = complement[(complement['outcome'] == 0)].shape[0] 
    vals = (tp, fp, tn, fn)
    conf_matrix(vals, 'Predicting Mortality on Entire Set')
    label = 'entire set'
    print(f'Accuracy on {label} = {accuracy(vals)*100:.0f}%')
    print(f'Recall on {label} = {recall(vals)*100:.0f}%')
    print(f'Precision on {label} = {precision(vals)*100:.0f}%')

# df = pd.read_csv('cardiacData.txt')
# for feature in df.columns[:-1]:
#     cor = df[feature].corr(df['outcome'])
#     print(f'Correlation of {feature} with outcome = {cor:.2f}')

# rule = lambda df: df[(df['numAttacks'] > 0)]
# test_rule(df, rule)
# sys.exit()

def scale_data(vals):
    """Assumes vals a sequence of numbers"""
    vals = np.array(vals)
    mean = sum(vals) / len(vals)
    sd = np.std(vals)
    return (vals - mean) / sd

def get_cardiac_data(scale = False):
    df = pd.read_csv('cardiacData.txt')
    if scale:
        df['heartRate'] = scale_data(df.heartRate)
        df['stElev'] = scale_data(df.stElev)
        df['age'] = scale_data(df.age)
        df['numAttacks'] = scale_data(df.numAttacks)
    # Split into features and label
    X = df[['heartRate','stElev','age','numAttacks']]
    Y = df.outcome
    return X, Y

def print_stats(label, vals_list):
    vals = []
    for i in range(4):
        vals.append(sum([vals[i] for vals in vals_list])/len(vals_list))
    print(f'Mean accuracy on {label} = {accuracy(vals)*100:.0f}%')
    print(f'Mean recall on {label} = {recall(vals)*100:.0f}%')
    try:
        print(f'Mean precision on {label} = {precision(vals)*100:.0f}%')
    except:
        print('No positive predictions')
        
def test_cardiac_model_RF(X,  Y, num_trees, max_depth, split = True,
                       cf = False, verbose = False):
    if split:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .4)
    
    else:
        X_train, X_test, Y_train, Y_test = X, X, Y, Y
    
    RF = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth)
    model = RF.fit(X_train, Y_train)
    test_vals = test_model(model, X_test, Y_test, verbose)
    if cf:
        conf_matrix(test_vals, 'Predicting Mortality on Test Set')
    train_vals = test_model(model, X_train, Y_train, verbose)
    if cf:
        conf_matrix(train_vals, 'Predicting Mortality on Training Set')
    return model, train_vals, test_vals

# feature_names = ('heartRate', 'stElev', 'age', 'numAttacks')
# set_seeds(0)

# # Look at 1 tree
# X, Y = get_cardiac_data()
# for depth in (12,):
#     model, train_vals, test_vals = test_cardiac_model_RF(X,  Y, 1, depth,
#                                                       cf = True, split = True)
#     estimator = model.estimators_[0]  # pick one tree from the forest
#     print(export_text(estimator, feature_names = feature_names))
#     print_stats('training set', [train_vals])
#     print_stats('test set', [test_vals])
    
# sys.exit()
   
def test_cardiac_model_Log(X,  Y, num_trees, max_depth,
                       cf = False, verbose = False):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .4)
    
    RF = RandomForestClassifier(
        n_estimators=num_trees,
        max_depth=max_depth,
        max_features = 'sqrt',
        bootstrap = True)
    model = RF.fit(X_train, Y_train)
    tp, fp, tn, fn = test_model(model, X_test, Y_test, verbose)
    test_vals = (tp, fp, tn, fn)
    if cf:
        conf_matrix(tp, fp, tn, fn, 'Predicting Mortality on Test Set')
    tp, fp, tn, fn = test_model(model, X_train, Y_train, verbose)
    train_vals = (tp, fp, tn, fn)
    if cf:
        conf_matrix(tp, fp, tn, fn, 'Predicting Mortality on Training Set')
    return model, train_vals, test_vals

def test_params_RF(X, Y, num_trees, max_depth, split = True, verbose = False):
    if verbose:
        print(f'num_trees = {num_trees}, max_depth = {max_depth}')
    model, train_vals, test_vals =\
         test_cardiac_model_RF(X, Y, num_trees, max_depth,
                            cf = False, verbose = False)
    return train_vals, test_vals

def run_trials_RF(X, Y, num_trials, num_trees, max_depth, split = True,
                  verbose = False):
    train_vals_list, test_vals_list = [], []           
    for _ in range(num_trials):
        train_vals, test_vals = test_params_RF(X, Y,num_trees, max_depth,
                                               split = split,
                                               verbose = verbose)
        train_vals_list.append(train_vals)
        test_vals_list.append(test_vals)
    print(f'{num_trials} trials of {num_trees} trees with max depth {max_depth}')
    print_stats('training', train_vals_list)
    print_stats('test', test_vals_list)
    
# Try random forest models
set_seeds(0)

X, Y = get_cardiac_data()
pos_frac = sum(Y)/len(Y)
print(f'Positive fraction in data set = {pos_frac:.2f}\n')
run_trials_RF(X, Y, 100, 1, 8, split = True, verbose = False)
run_trials_RF(X, Y, 100, 100, 8, split = True, verbose = False)

sys.exit()