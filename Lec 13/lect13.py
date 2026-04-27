#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:30:38 2025

@author: johnguttag
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
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
plt.rcParams['lines.markersize'] = 4


## code used in lecture 12

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
    
def set_seeds(val):
    random.seed(val)
    np.random.seed(val)
       
def test_model(forest, X, Y, verbose = False):
    y_hats = forest.predict(X)
    vals = confusion_matrix(Y, y_hats).ravel()
    return vals

# def test_rule(df, rule):
#     subset = rule(df)
#     tp = subset[(subset['outcome'] == 1)].shape[0]
#     fp = subset[(subset['outcome'] == 0)].shape[0]
#     complement = df.loc[~df.index.isin(subset.index)]
#     fn = complement[(complement['outcome'] == 1)].shape[0]
#     tn = complement[(complement['outcome'] == 0)].shape[0] 
#     vals = (tp, fp, tn, fn)
#     conf_matrix(vals, 'Predicting Mortality on Entire Set')
#     label = 'entire set'
#     print(f'Accuracy on {label} = {accuracy(vals)*100:.0f}%')
#     print(f'Recall on {label} = {recall(vals)*100:.0f}%')
#     print(f'Precision on {label} = {precision(vals)*100:.0f}%')

## Change get_cardiac data so that it can scale featues

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

## Code new to Lecture 13

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
    
def test_base(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .4)
    Y_test = Y_test.values.tolist()
    test_preds = [0 for _ in range(len(X_test))]
    test_vals = confusion_matrix(Y_test, test_preds).ravel()
    return test_vals

def test_random(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .4)
    Y_test = Y_test.values.tolist()
    train_prev = sum(Y_train)/len(Y_train)
    test_preds = [1 if random.random() <= train_prev else 0
                  for _ in range(len(X_test))]
    test_vals = confusion_matrix(Y_test, test_preds).ravel()
    return test_vals
    
def run_trials_base(X, Y, num_trials):
    test_vals_list = []          
    for _ in range(num_trials):
        test_vals_list.append(test_base(X, Y))
    print(f'For {num_trials} trials of majority class baseline')
    print_stats('test', test_vals_list)
    
def run_trials_random(X, Y, num_trials):
    set_seeds(0)
    test_vals_list = []          
    for _ in range(num_trials):
        test_vals_list.append(test_random(X, Y))
    print(f'For {num_trials} trials of weighted random baseline')
    print_stats('test', test_vals_list)

feature_names = ('heartRate', 'stElev', 'age', 'numAttacks')
set_seeds(0)

# # Look at 1 tree
# X, Y = get_cardiac_data()
# model, _, _ = test_cardiac_model_RF(X,  Y, 1, 3, cf = False)
# for tree_num in range(1):
#     estimator = model.estimators_[tree_num]  # pick one tree from the forest
#     print(export_text(estimator, feature_names = feature_names))

# sys.exit()

# # Compare 1 decision tree of different depths to baseline
# X, Y = get_cardiac_data()
# pos_frac = sum(Y)/len(Y)
# print(f'Positive fraction in data set = {pos_frac:.2f}\n')
# run_trials_RF(X, Y, 100, 1, 1, split = False, verbose = False)
# run_trials_RF(X, Y, 100, 1, 2, split = False, verbose = False)
# run_trials_RF(X, Y, 100, 1, 3, split = False, verbose = False)
# run_trials_RF(X, Y, 100, 1, 12, split = False, verbose = False)
# run_trials_base(X, Y, 100)
# run_trials_random(X, Y, 100)

# sys.exit()

# Try random forest models

# X, Y = get_cardiac_data()
# pos_frac = sum(Y)/len(Y)
# print(f'Positive fraction in data set = {pos_frac:.2f}\n')
# run_trials_RF(X, Y, 100, 1, 8, split = True, verbose = False)
# run_trials_RF(X, Y, 100, 100, 8, split = True, verbose = False)

# run_trials_base(X, Y, 100)
# run_trials_random(X, Y, 100)

# sys.exit()

def test_cardiac_model_AB(X,  Y, max_depth,
                       split = True, cf = False, verbose = False):
    if split:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .4)
    
    else:
        X_train, X_test, Y_train, Y_test = X, X, Y, Y
    
    
    base = DecisionTreeClassifier(max_depth = max_depth)
    AB = AdaBoostClassifier(estimator=base, n_estimators=50,
                                    learning_rate=1.0, random_state=0)
    model = AB.fit(X_train, Y_train)
    test_vals = test_model(model, X_test, Y_test, verbose)
    if cf:
        conf_matrix(test_vals, 'Predicting Mortality on Test Set')
    train_vals = test_model(model, X_train, Y_train, verbose)
    if cf:
        conf_matrix(train_vals, 'Predicting Mortality on Training Set')
    return train_vals, test_vals

def run_trials_AB(X, Y, max_depth, num_trials):
    train_vals_list, test_vals_list = [], []           
    for _ in range(num_trials):
        train_vals, test_vals = test_cardiac_model_AB(X, Y, max_depth)
        train_vals_list.append(train_vals)
        test_vals_list.append(test_vals)
    print(f'{num_trials} trials of AdaBoost with {max_depth}-level decision tree')
    print_stats('training', train_vals_list)
    print_stats('test', test_vals_list)

# # Data
# X, Y = get_cardiac_data()
# run_trials_AB(X, Y, 4, 100)

# sys.exit()

## Produce plots for slides

def sigmoid(z):
    return 1/(1+np.e**-(z))

def compute_grad(X, y, w):
    """
    Compute gradient of cross entropy function with sigmoidal probabilities

    Args: 
        X (numpy.ndarray): examples. Individuals in rows, features in columns
        y (numpy.ndarray): labels. Vector corresponding to rows in X
        w (numpy.ndarray): weight vector

    Returns: 
        numpy.ndarray 

    """
    m = X.shape[0]
    Z = w.dot(X.T)
    A = sigmoid(Z)
    out=  (-1/ m) * (X.T * (A - y)).sum(axis=1) 

    return out

def make_pred(weights, xs, intercept = 0):
    if (len(weights) == 1):
        Z = weights * xs + intercept
    else: 
        Z = weights.dot(xs.T) + intercept
    A = sigmoid(Z) #prediction
    return A

def make_heights(n1, n2):
    men_mean = 69
    women_mean = 63
    std = 3

    men = np.random.normal(loc = men_mean, scale = std,size = n1) 
    women = np.random.normal(loc = women_mean, scale = std, size = n2) 

    labels = np.concatenate([np.ones(n1), np.zeros(n2)])
    return np.concatenate([men, women]), labels

def make_heights_plot():
    random.seed(0)
    points, labels = make_heights(50, 50)
    ones = np.ones(len(points))

    plt.scatter(points, labels)
    plt.ylabel("Sex")
    plt.xlabel("Height")
    plt.title('Height vs. Sex')
    plt.figure()

    points, labels = make_heights(50, 50)  #initialize heights
    ones = np.ones(len(points)) 

    # data = points / sd
    data = np.dstack([points, ones])[0]  #add a  vector of ones for intercept
    weights = np.array([1, 1])
    rate = 0.01
    for i in range(1_000_000):
        grad = compute_grad(data, labels, weights)
        weights = weights + grad * rate

    xs = np.arange(50, 80, 0.1)
    ones = np.ones(len(xs))

    plt.scatter(points, labels)
    plt.ylabel("Sex")
    plt.xlabel("Height")
    plt.title('Height vs. Sex')
    preds = make_pred(weights, np.dstack([xs,ones])[0]) 
    preds = preds
    plt.plot(xs,preds, color='r')
    plt.figure()

    linearR = LinearRegression(fit_intercept=True)
    d = np.reshape(points,(-1,1))
    linearR.fit(d, labels)
    ypred = linearR.predict(np.reshape(xs,(-1,1)))
    plt.scatter(points, labels)
    plt.ylabel("Sex")
    plt.xlabel("Height")
    plt.title('Height vs. Sex')
    plt.plot(xs,ypred, color='r')
    plt.show()

# make_heights_plot() 
# sys.exit()

# Test a logistic regression model

def test_cardiac_model_LR(X, Y, cf=False, verbose=False):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
    LR = LogisticRegression(max_iter=1000, class_weight = 'balanced')
    model = LR.fit(X_train, Y_train)
    test_vals = test_model(model, X_test, Y_test, verbose)
    if cf:
        conf_matrix(test_vals, 'Predicting Mortality on Test Set')
    train_vals = test_model(model, X_train, Y_train, verbose)
    if cf:
        conf_matrix(train_vals, 'Predicting Mortality on Training Set')
    return train_vals, test_vals, model

def run_trials_LR(X, Y, num_trials, verbose = False):
    train_vals_list, test_vals_list, coefs_list = [], [], []          
    for _ in range(num_trials):
        train_vals, test_vals, model = test_cardiac_model_LR(X, Y)
        train_vals_list.append(train_vals)
        test_vals_list.append(test_vals)
        coefs_list.append(model.coef_[0])
    print_stats('training', train_vals_list)
    print_stats('test', test_vals_list)
    mean_coefs = []
    for i in range(4):
        mean_coefs.append(sum([coef[i] for coef in coefs_list])/len(coefs_list))
    print(f'Mean coefficients: HR = {mean_coefs[0]:.3f}, stElev = {mean_coefs[1]:.3f}, '
          f'Age = {mean_coefs[2]:.3f}, prevAttacks = {mean_coefs[3]:.3f}')

# X, Y = get_cardiac_data(scale = True)
# set_seeds(0)
# print('Try 100 trials of a balanced logistic regression')
# run_trials_LR(X, Y, 100)

# sys.exit()

## Clustering

############################################################
# General feature represention and pairwise plotting
############################################################


def minkowski_dist(v1, v2, p):
    """Assumes v1 and v2 are equal length arrays of numbers"""
    dist = 0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])**p
    return dist**(1/p)


class Example:

    def __init__(self, name, features, feature_names, label=None):
        """Assumes features is an array of floats"""
        self.name = name
        self.features = features
        self.feature_names = feature_names
        self.label = label

    def get_name(self):
        return self.name

    def get_features(self):
        return self.features.copy()

    def get_feature_names(self):
        return self.feature_names.copy()

    def get_label(self):
        return self.label

    def dimensionality(self):
        return len(self.features)

    def distance(self, other):
        return minkowski_dist(self.features, other.features, 2)

    def __str__(self):
        return f"{self.name}:{self.features}:{self.label}"


############################################################
# Load patient data
############################################################


class Patient(Example):
    pass


def get_data(scale=False):
    df = pd.read_csv('cardiacData.txt') # Use Pandas to load example data
    print(f"num_positives = {df.outcome.sum()} out of {len(df)}")
    if scale:
        df['heartRate'] = scale_data(df.heartRate)
        df['stElev'] = scale_data(df.stElev)
        df['age'] = scale_data(df.age)
        df['numAttacks'] = scale_data(df.numAttacks)
    points = []
    for i, r in df.iterrows():
        features = np.array([r['heartRate'], r['age'], r['stElev'], r['numAttacks']])
        points.append(Patient(str(i), features,
                              ['heartRate', 'age', 'stElev', 'numAttacks'], r['outcome']))
    return points


def get_data_two_features(scale=False):
    """Just load two features (age and numAttacks)"""
    df = pd.read_csv('cardiacData.txt')
    print(f"num_positives = {df.outcome.sum()} out of {len(df)}")
    if scale:
        df['age'] = scale_data(df.age)
        df['numAttacks'] = scale_data(df.numAttacks)
    points = []
    for i, r in df.iterrows():
        features = np.array([r['age'], r['numAttacks']])
        points.append(Patient(str(i), features,
                              ['age', 'numAttacks'], r['outcome']))
    return points


def plot_features_pairwise(examples):
    """Make pairwise feature plots from a set of examples"""
    d = len(examples[0].features)
    _, axs = plt.subplots(int(d/2), d-1)
    axs = axs.flatten()
    cur = 0
    for i in range(d):
        for j in range(i, d):
            if i == j:
                continue

            x_pos, y_pos, x_neg, y_neg = [], [], [], []
            for e in examples:
                if e.get_label() == 1:
                    x_pos.append(e.features[i])
                    y_pos.append(e.features[j])
                else:
                    x_neg.append(e.features[i])
                    y_neg.append(e.features[j])

            ax = axs[cur]
            cur += 1
            ax.scatter(x_neg, y_neg, c='g', marker='.', label='live')
            ax.scatter(x_pos, y_pos, c='r', marker='d', label='die')
            ax.set_xlabel(examples[0].get_feature_names()[i])
            ax.set_ylabel(examples[0].get_feature_names()[j])
            ax.legend()

############################################################
# Cluster representation and plotting
############################################################


class Cluster:

    def __init__(self, examples):
        """Assumes examples a non-empty list of Examples"""
        self.examples = examples
        self.centroid = self.compute_centroid()

    def members(self):
        for e in self.examples:
            yield e

    def get_centroid(self):
        return self.centroid

    def compute_centroid(self):
        vals = np.array([0.0] * self.examples[0].dimensionality())
        for e in self.examples: # compute mean
            vals += e.get_features()
        centroid = Example('centroid', vals / len(self.examples),
                           self.examples[0].get_feature_names())
        return centroid

    def update(self, examples):
        """Assumes examples is a non-empty list of Examples
           Replace examples; return amount centroid has changed"""
        old_centroid = self.centroid
        self.__init__(examples)
        return old_centroid.distance(self.centroid)

    def inertia(self):
        tot_dist = 0
        for e in self.examples:
            tot_dist += e.distance(self.centroid)**2
        return tot_dist

    def __str__(self):
        names = []
        for e in self.examples:
            names.append(e.get_name())
        names.sort()
        result = f"Cluster with centroid {self.centroid.get_features()} contains:\n  "
        for e in names:
            result += f"{e}, "
        return result[:-2] # remove trailing comma and space

def total_inertia(clusters):
    """Assumes clusters a list of clusters
       Return a measure of the total dissimilarity of the clusters in the list"""
    tot_dist = 0
    for c in clusters:
        tot_dist += c.inertia()
    return tot_dist

def plot_clusters(clusters, xi=0, yi=1):
    """Plot the data and clusters"""
    plt.figure()
    colors = ['r', 'g', 'b', 'm', 'c']
    i = 0
    for c in clusters:
        col = colors[i % len(colors)]
        i += 1
        x_pos, y_pos, x_neg, y_neg = [], [], [], []

        # Just collect the two features specified by indices
        for p in c.members():
            if p.get_label() == 1:
                x_pos.append(p.features[xi])
                y_pos.append(p.features[yi])
            else:
                x_neg.append(p.features[xi])
                y_neg.append(p.features[yi])

        # Plot cluster data with outcomes
        plt.scatter(x_pos, y_pos, c=col, marker='d', label=f"{i}-die")
        plt.scatter(x_neg, y_neg, c=col, marker='.', label=f"{i}-live")

        # Plot cluster centroid
        plt.scatter([c.get_centroid().features[xi]], [[c.get_centroid().features[yi]]],
                    c=col, marker='*', s=150)
        plt.text(x=c.get_centroid().features[xi]+0.1, y=c.get_centroid().features[yi]+0.05,
                 s=f"Centroid {i}", fontsize=12)
        plt.xlabel(c.get_centroid().get_feature_names()[xi])
        plt.ylabel(c.get_centroid().get_feature_names()[yi])
    plt.legend()
    plt.show()


############################################################
# K-means clustering
############################################################


def kmeans(examples, k, verbose=False):
    # Get k randomly chosen initial centroids, create cluster for each
    initial_centroids = random.sample(examples, k)
    clusters = []
    for e in initial_centroids:
        clusters.append(Cluster([e]))

    # Iterate until centroids do not change
    converged = False
    num_iterations = 0
    while not converged:
        num_iterations += 1
        # Initialize k new clusters as distinct empty lists
        new_clusters = []
        for i in range(k):
            new_clusters.append([])

        # Associate each example with closest centroid
        for e in examples:
            # Find the centroid closest to e
            smallest_distance = e.distance(clusters[0].get_centroid())
            index = 0
            for i in range(1, k):
                distance = e.distance(clusters[i].get_centroid())
                if distance < smallest_distance:
                    smallest_distance = distance
                    index = i
            # Add e to the list of examples for appropriate cluster
            new_clusters[index].append(e)

        for c in new_clusters: # Avoid having empty clusters
            if len(c) == 0:
                raise ValueError("Empty Cluster")

        # Update each cluster; check if a centroid has changed
        converged = True
        for i in range(k):
            if clusters[i].update(new_clusters[i]) > 0:
                converged = False

        if verbose:
            print(f"Iteration #{num_iterations}")
            for c in clusters:
                print(c)
            print() # Add blank line
            # plot_clusters(clusters)

    return clusters


def try_kmeans(examples, num_clusters, num_trials, verbose=False):
    """Call kmeans num_trials times and return the result with the lowest dissimilarity"""
    best = None
    for _ in range(num_trials):
        while True: # Keep trying until no empty clusters
            try:
                clusters = kmeans(examples, num_clusters, verbose)
                break
            except ValueError:
                continue
        if best is None or total_inertia(clusters) < total_inertia(best):
            best = clusters
    return best


############################################################
# Perform clustering on cardiac data
############################################################


# Find and plot clusters for two features

# patients = get_data_two_features()
# best_clustering = try_kmeans(patients, num_clusters=2, num_trials=5)
# plot_clusters(best_clustering)

# sys.exit()


# # Regenerate clusters under feature scaling
# patients = get_data_two_features(scale=True)
# best_clustering = try_kmeans(patients, num_clusters=2, num_trials=5)
# plot_clusters(best_clustering)

# sys.exit()

############################################################
# Evaluate goodness of clustering as we allow more clusters
############################################################


def score_clustering(clusters):
    """Assumes clusters is a sequence of clusters
       Return array of tuples of (number of positive examples, total number of points, fraction positives)"""
    scores = []
    for c in clusters:
        num_pts = 0
        num_pos = 0
        for p in c.members():
            num_pts += 1
            if p.get_label() == 1:
                num_pos += 1
        frac_pos = num_pos / num_pts
        scores.append((num_pos, num_pts, frac_pos))
    return scores

def get_clustering_stats(clusters, labels=None, verbose=False):
    """Print information about each cluster
       * clusters is a sequence of clusters
       * labels is a set of labels for each cluster (True = positive)
         If None, labels are computed based on whether fraction of positives in cluster > 0.5"""
    scores = score_clustering(clusters)
    total = 0
    true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
    total_correct = 0
    if labels == None:
        labels = [frac_pos > 0.5 for _, _, frac_pos in scores]
    for i in range(len(scores)):
        num_pos, num_pts, frac_pos = scores[i]
        if labels[i]:
            guess = 'Positive'
        else:
            guess = 'Negative'
        print(f"Cluster of size {num_pts}, "
              f"frac. pos. = {frac_pos:.3f}, num. pos. = {num_pos}, Guess = {guess}")
        total += num_pts
        if labels[i]:
            true_pos += num_pos
            false_pos += num_pts - num_pos
            total_correct += num_pos
        else:
            true_neg += num_pts - num_pos
            false_neg += num_pos
            total_correct += num_pts - num_pos
    sensitivity = true_pos / (true_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)

    if verbose:
        print(f"Overall Total correct = {total_correct} out of {total} ({total_correct/total:.3f})")
        # print(f"Sensitivity = {sensitivity:.3f}")
        # print(f"Specificity = {specificity:.3f}")
        # print(f"               Pred=1    Pred=0")
        # print(f"             ----------------------")
        # print(f"Actual = 1   | {true_pos} \t {false_neg}")
        # print(f"Actual = 0   | {false_pos} \t {true_neg}")
    return sensitivity, specificity, total_correct, total


def eval_num_clusters(patients, num_clusters, num_trials=5):
    best_clustering = try_kmeans(patients, num_clusters, num_trials)
    # plot_clusters(best_clustering, xi=2, yi=3) # Plot clusters in the space of stElev and numAttacks
    stats = get_clustering_stats(best_clustering, verbose = False)
    return stats


# # Increasing k results in better and better clusters
# patients = get_data(scale=True)
# for k in 2, 4, 6, 50:
#     print('=' * 60)
#     print(f"Test k-means with k={k} clusters")
#     eval_num_clusters(patients, num_clusters=k)
#     print()


############################################################
# Prevent overfitting by cross-validating on test data
############################################################


def eval_on_test_data(clusters, test_data):
    # Compute the labels based on the initial clustering
    old_scores = score_clustering(clusters)
    old_labels = [frac_pos > 0.5 for _, _, frac_pos in old_scores]

    # Assign each test point to the nearest cluster
    new_clusters = [[] for c in clusters]
    for p in test_data:
        min_dist = p.distance(clusters[0].get_centroid())
        best_cluster = 0
        for c in range(1, len(clusters)):
            center = clusters[c].get_centroid()
            dist = p.distance(center)
            if dist < min_dist:
                best_cluster = c
                min_dist = dist
        new_clusters[best_cluster].append(p)

    # Discard empty clusters
    new_cluster_objects = []
    filtered_labels = []
    for i in range(len(new_clusters)):
        if len(new_clusters[i]) > 0:
            new_cluster_objects.append(Cluster(new_clusters[i]))
            filtered_labels.append(old_labels[i])

    # Print some stats about how accurate this clustering is
    stats = get_clustering_stats(new_cluster_objects, labels=filtered_labels)
    return stats


def bootstrap(v, stat_f, percentiles=[2.5, 97.5], n_iters=100):
    """Estimate confidence interval of a statistic on a population,
       given a single sample from that population
       * v is the sample
       * stat_f is a function that evaluates the statistic on the population,
           or a subset thereof"""
    ar = []
    v_np = np.array(v)
    # print(v_np)
    for i in range(n_iters):
        ind_samp = np.random.choice(v_np.shape[0], size=len(v))
        if len(v_np.shape) == 1:
            ar.append(stat_f(v_np[ind_samp]))
        else:
            ar.append(stat_f(v_np[ind_samp,:]))
    return np.percentile(ar, percentiles)


if True:
    patients = get_data(scale=True)

    ks = 2, 4, 6, 10, 20, 50
    tot_sens = [[] for k in ks]
    tot_spec = [[] for k in ks]
    tots = [[] for k in ks]
    tot_corrects = [[] for k in ks]

    folds = 10
    for fold in range(folds): # n-fold cross validation
        train_patients, test_patients = [], []
        for p in patients:
            if random.random() > 0.4: # use 40% of data for testing
                train_patients.append(p)
            else:
                test_patients.append(p)

        for i in range(len(ks)):
            k = ks[i]
            print(f"pass {fold}, trying k = {k}")
            best_clustering = try_kmeans(train_patients, k, 5)
            # plot_clusters(best_clustering)
            sens, spec, tot_correct, tot = eval_on_test_data(best_clustering, test_patients)
            tot_sens[i].append(sens)
            tot_spec[i].append(spec)
            tots[i].append(tot)
            tot_corrects[i].append(tot_correct)


    # Print out specificity and sensitivity
    # Also include confidence estimates using bootstrap method (not covered in lecture)
    for i in range(len(ks)):
        print()

        conf_lo, conf_hi = bootstrap(list(zip(tot_corrects[i], tots[i])),
                                      lambda x: sum(x[:,0]) / sum(x[:,1]))
        print(f"k={ks[i]}, accuracy = {sum(tot_corrects[i]) / sum(tots[i]):.3f}, "
                f"95% conf = [{conf_lo:.3f}, {conf_hi:.3f}]")

        conf_lo, conf_hi = bootstrap(tot_sens[i], np.mean)
        print(f"k={ks[i]}, mean recall = {np.mean(tot_sens[i]):.3f}, "
              f"95% conf = [{conf_lo:.3f}, {conf_hi:.3f}]")

        conf_lo, conf_hi = bootstrap(tot_spec[i], np.mean)
        print(f"k={ks[i]}, mean precision = {np.mean(tot_spec[i]):.3f}, "
              f"95% conf = [{conf_lo:.3f}, {conf_hi:.3f}]")


    # Finally plot overall performance with k=2 and k=4
    patients = get_data(scale=True)
    for k in 2, 4:
        best_clustering = try_kmeans(patients, k, 5)
        # plot_clusters(best_clustering)


############################################################
# Show all plots
############################################################


plt.show()
