### You are given this code file to study before the quiz. ###
### The only new thing you have not seen before is a new implementation of kmeans ###
### Understand this code and make small changes to it ##
## During the quiz, you'll be asked some questions about it ##

import random
import numpy as np

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
        return result[:-2] 

## New implementation of kmeans, not seen before in class ##

def kmeans(examples, k, num_iters=5):
    centroids = random.sample(examples, k)
    centroids_history = []
    labels_history = []

    for iteration in range(num_iters):
        clusters = [[] for _ in range(k)]
        labels = []
        for e in examples:
            smallest_distance = e.distance(centroids[0])
            index = 0
            for i in range(1, k):
                distance = e.distance(centroids[i])
                if distance < smallest_distance:
                    smallest_distance = distance
                    index = i
            clusters[index].append(e)
            labels.append(index)
        labels_history.append(labels)

        for c in clusters:
            if len(c) == 0:
                raise ValueError("Empty cluster encountered")

        new_centroids = []
        for cluster in clusters:
            new_cluster = Cluster(cluster)
            new_centroids.append(new_cluster.get_centroid())
        centroids_history.append(new_centroids)

        print(f"Iteration #{iteration + 1}")
        for i in range(k):
            shift = round(centroids[i].distance(new_centroids[i]), 2)
            print(f" Centroid {i}: {centroids[i]} -> {new_centroids[i]} shift={shift}")

        centroids = new_centroids

    final_clusters = [[] for _ in range(k)]
    for e in examples:
        smallest_distance = e.distance(centroids[0])
        index = 0
        for i in range(1, k):
            distance = e.distance(centroids[i])
            if distance < smallest_distance:
                smallest_distance = distance
                index = i
        final_clusters[index].append(e)

    for i in range(len(final_clusters)):
        final_clusters[i] = Cluster(final_clusters[i])

    return centroids, final_clusters, centroids_history, labels_history



feature_names = ["x", "y"]
examples = [
    Example("A", np.array([1.0, 1.0]), feature_names),
    Example("B", np.array([1.2, 1.1]), feature_names),
    Example("C", np.array([0.8, 0.9]), feature_names),
    
    Example("D", np.array([5.0, 5.0]), feature_names),
    Example("E", np.array([5.3, 5.1]), feature_names),
    Example("F", np.array([4.9, 4.8]), feature_names),
]
for c in kmeans(examples, 2, 5)[1]:
    print(c)

