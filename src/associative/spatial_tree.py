# Spatial trees can be used to efficiently query multidimensional data.
# Great paper summarizing what spatial trees are:
# Multidimensional binary search trees used for associative searching: https://brianmcfee.net/papers/ismir2011_sptree.pdf
# Original implementation taken from: https://github.com/bmcfee/spatialtree

import unittest

import numpy as np
import scipy.stats
import random
import heapq

from enum import Enum

# Enumeration representing the split rule we want to use. Can be one of either:
#   KD_TREE:  Standard KD-tree: can be effective for low-dimensional data, but performs poorly in higher dimensions.
#   PCA_TREE: Chooses the split direction which maximizes the variance. This rule may be more effective than KD-tree at
#             reducing the variance at each split in the tree.
#   K_MEANS:  The k-means rule produces splits which attempt preserve cluster structure. Uses k-means algorithm.
#   RANDOM_PROJECTION: Simply takes a direction uniformly at random. This rule is simple to compute and adapts to
#                      the intrinsic dimensionality of the data.
SplitRule = Enum('SplitRule', ['KD_TREE', 'PCA_TREE', 'K_MEANS', 'RANDOM_PROJECTION'])

# Minimum number of steps for building the k-means tree
MIN_STEPS_K_MEANS = 1000
# Number of directions to consider for each random projection split
SAMPLES_FOR_RANDOM_PROJECTION = 10

class SpatialTree(object):

    def __init__(self
                 # A data matrix with one point per row or a dict of vectors containing the data
                 , data
                 # This is the splitting rule we want to use (KD_TREE, PCA_TREE, K_MEANS or RANDOM_PROJECTION)
                 , split_rule=SplitRule.KD_TREE
                 # The fraction of the data that should propagate to both children during splits.
                 , spill_fraction=0.25
                 # The minimum number of items required to split a node
                 , min_items_for_split=64
                 # List of keys/indices to store in this (sub)tree
                 , indices=()
                 , height = None
                 ):

        assert 0 <= spill_fraction and spill_fraction < 1

        if not indices:
            if isinstance(data, dict):
                indices = list(data.keys())
            else:
                indices = list(range(len(data)))

        num_indices = len(indices)

        height = max(0, int(
            np.ceil(
                np.log(num_indices / 500)
                / np.log(2.0 / (1 + spill_fraction))
            )
        )) if height is None else height

        self.indices = set(indices)
        self.split_rule = split_rule
        self.spill_fraction = spill_fraction
        self.children = None
        self.split_direction_vector = None
        self.thresholds = None
        self.is_key_value_store = isinstance(data, dict)

        # Compute the dimensionality of the data
        for x in self.indices:
            self.num_dimensions = len(data[x])
            break

        # Split the new node
        self.height = self.split(data, split_rule, min_items_for_split, height, indices)

    # Split the tree into two children based on the given split rule and data.
    def split(self, data, split_rule, min_items_for_split, height, indices):

        # First, find the split rule
        if split_rule == SplitRule.KD_TREE:
            split_function = self.kd_tree
        elif split_rule == SplitRule.PCA_TREE:
            split_function = self.pca_tree
        elif split_rule == SplitRule.K_MEANS:
            split_function = self.two_means
        elif split_rule == SplitRule.RANDOM_PROJECTION:
            split_function = self.random_projection
        else:
            raise ValueError(f'Unsupported split rule: {split_rule}.')

        # If the height is 0, or the set is too small, then we don't need to split
        if height == 0 or len(indices) < min_items_for_split:
            return 0

        # Compute the split direction
        self.split_direction_vector = split_function(data)

        # Project onto split direction
        split_direction_projections = {}

        for idx in self.indices:
            split_direction_projections[idx] = np.dot(self.split_direction_vector, data[idx])

        # Compute the bias points
        self.thresholds = scipy.stats.mstats.mquantiles(
            list(split_direction_projections.values())
            , [0.5 - self.spill_fraction / 2, 0.5 + self.spill_fraction / 2]
        )

        # Partition the data
        left_set = set()
        right_set = set()

        for (idx, value) in split_direction_projections.items():
            if value >= self.thresholds[0]:
                right_set.add(idx)
            if value < self.thresholds[-1]:
                left_set.add(idx)

        # Clean up split_direction_projections as it's no longer needed
        del split_direction_projections

        # Construct the children
        self.children = [None] * 2

        self.children[0] = SpatialTree(
            data
            , split_rule=split_rule
            , spill_fraction=self.spill_fraction
            , min_items_for_split = min_items_for_split
            , indices = left_set
            , height=height-1
        )

        self.children[1] = SpatialTree(
            data
            , split_rule=split_rule
            , spill_fraction=self.spill_fraction
            , min_items_for_split=min_items_for_split
            , indices=right_set
            , height=height-1
        )

        tree_height = 1 + max(self.children[0].height, self.children[1].height)

        return tree_height

    # Add new data to the tree.  Note: this does not rebalance or split the tree.
    def update(self, data_dict):

        assert self.is_key_value_store

        # Update the node's indices with new keys
        self.indices.update(list(data_dict.keys()))

        if self.is_leaf():
            # If the node is a leaf, we stop updating as it doesn't have any children.
            return

        # Create left and right sets to hold data points based on the split direction and threshold
        left_set = {}
        right_set = {}

        # Iterate through the data_dict and divide the data into left and right sets
        for key, vector in data_dict.items():
            split_direction_similarity = np.dot(self.split_direction_vector, vector)

            if split_direction_similarity >= self.thresholds[0]:
                # Add the data to the right set if the similarity is greater than or equal to the right threshold
                right_set[key] = vector

            if split_direction_similarity < self.thresholds[-1]:
                # Add the data to the left set if the similarity is less than the left threshold
                left_set[key] = vector

        # Recursively update the left and right children nodes with the corresponding data sets
        self.children[0].update(left_set)
        self.children[1].update(right_set)

    # Returns the split rule for this node: a tuple (w, (lower_threshold, upper_threshold))
    # where w is a vector, and the thresholds are scalars.
    def node_split_rule(self):
        return (self.split_direction_vector, self.thresholds)

    def is_leaf(self):
        return self.height == 0

    def __len__(self):
        return len(self.indices)

    def __contains__(self, item):
        return item in self.indices

    def __iter__(self):
        return self.indices.__iter__()

    # Iterator for in-order transversal within our tree
    def traverse(self):
        if self.is_leaf():
            # If the current node is a leaf, yield itself
            yield self
        else:
            # Traverse the left subtree
            for child in self.children[0].traverse():
                yield child

            # Yield the current node
            yield self

            # Traverse the right subtree
            for child in self.children[1].traverse():
                yield child

    # Removes the specified item from the node and its descendants (subtree).
    def remove(self, item):
        if item not in self.indices:
            # If the item is not in this node's indices, raise a KeyError
            raise KeyError(item)

        if not self.is_leaf():
            # If the node is not a leaf, check each child for the item and remove it recursively
            for child in self.children:
                if item in child:
                    child.remove(item)

        # Remove the item from this node's indices
        self.indices.remove(item)

    # Compute the retrieval set for either a given query index or vector.
    # Exactly one of index or data vector must be supplied.
    def retrieval_set(self, **kwargs):

        # Ensure that we are either supplying an index or a vector as part of the keyword arguments
        assert 'index' in kwargs or 'vector' in kwargs
        # If an index is supplied, ensure that it is present within this node
        assert kwargs['index'] in self if 'index' in kwargs else True

        # Helper function to compute the retrieval set from a given query index.
        def retrieve_from_index(idx):

            retrieval_set = set()

            if idx in self.indices:
                if self.is_leaf():
                    retrieval_set = self.indices.difference([idx])
                else:
                    for child in self.children:
                        if idx in child:
                            retrieval_set |= child.retrieval_set(index=idx)

            return retrieval_set

        # Helper function to compute the retrieval set from a given data vector.
        def retrieve_from_vector(vector):

            retrieval_set = set()

            if self.is_leaf():
                # If we landed on the leaf, we are done
                retrieval_set = self.indices
            else:
                split_direction_similarity = np.dot(self.split_direction_vector, vector)

                if split_direction_similarity >= self.thresholds[0]:
                    retrieval_set |= self.children[1].retrieval_set(vector=vector)

                if split_direction_similarity < self.thresholds[-1]:
                    retrieval_set |= self.children[0].retrieval_set(vector=vector)

            return retrieval_set

        if 'index' in kwargs:
            return retrieve_from_index(kwargs['index'])
        elif 'vector' in kwargs:
            return retrieve_from_vector(kwargs['vector'])
        else:
            raise ValueError('SpatialTree.retrieval_set must be supplied with either an index or a data vector.')

    # Returns a sorted list of the indices of k-nearest (approximate) neighbors of the query.
    def k_nearest_neighbors(self, data, **kwargs):

        # Validate 'k' parameter
        k = kwargs.get('k')

        # Validate that parameter k is supplied and that it is an integer greater than or equal to 1
        assert k is not None and isinstance(k, int) and k >= 1

        # Get the query point or vector
        if 'index' in kwargs:
            query_point = data[kwargs['index']]
        else:
            query_point = kwargs['vector']

        # Compute distances from the query point to the retrieval set
        def calculate_distance(retrieval_set):
            for idx in retrieval_set:
                yield (np.sum((query_point - data[idx]) ** 2), idx)

        # Get indices in sorted order
        retrieval_set = self.retrieval_set(**kwargs)

        nearest_neighbors = [idx for (distance, idx) in heapq.nsmallest(k, calculate_distance(retrieval_set))]
        return nearest_neighbors

    # Prune the tree such that the height <= max_height.
    def prune(self, max_height):

        assert isinstance(max_height, int) and max_height >= 0

        # If we're already a leaf, nothing to do
        if self.height == 0:
            return

        # If max_height is 0, prune here
        if max_height == 0:
            self.height = 0
            self.split_direction_vector = None
            self.children = None
            self.thresholds = None
            return

        # Otherwise, recursively prune
        self.children[0].prune(max_height - 1)
        self.children[1].prune(max_height - 1)
        self.height = 1 + max(self.children[0].height(), self.children[1].height())

    # Computes a split direction by the top principal component (leading eigenvector
    # of the covariance matrix) of data in the current node.
    def pca_tree(self, data):

        # Compute the first moment (mean)
        first_moment = np.zeros(self.num_dimensions)

        # Compute the second moment (covariance matrix)
        second_moment = np.zeros((self.num_dimensions, self.num_dimensions))

        # Calculate the first and second moments
        for i in self.indices:
            first_moment += data[i]
            second_moment += np.outer(data[i], data[i])

        # Calculate the mean
        mean = first_moment / len(self)

        # Calculate the covariance matrix
        covariance = (second_moment - (len(self) * np.outer(mean, mean))) / (len(self) - 1.0)

        # Perform eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Select the top eigenvector
        top_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

        return top_eigenvector

    # Finds the coordinate axis with the highest variance.
    def kd_tree(self, data):

        num_dimensions = self.num_dimensions  # Number of dimensions in the data points
        num_data_points = len(self)  # Total number of data points

        moment_1 = np.zeros(num_dimensions)  # First moment (sum of coordinates)
        moment_2 = np.zeros(num_dimensions)  # Second moment (sum of squares of coordinates)

        # Calculate the first and second moments for each dimension
        for idx in self.indices:
            moment_1 += data[idx]
            moment_2 += data[idx] ** 2

        # Calculate the mean for each dimension
        moment_1 /= num_data_points

        # Calculate the variance for each dimension
        sigma = (moment_2 - (num_data_points * moment_1 ** 2)) / (num_data_points - 1.0)

        # Find the coordinate axis with the highest variance
        split_direction_vector = np.zeros(num_dimensions)
        split_direction_vector[np.argmax(sigma)] = 1

        return split_direction_vector

    # Computes a split direction by clustering the data in the current node
    # into two, and choosing the direction spanned by the cluster centroids
    # The cluster centroids are found by an online k-means with the Hartigan
    # update.  The algorithm runs through the data in random order until
    # a specified minimum number of updates have occurred.
    def two_means(self, data):

        def euclidean_distance(v1, v2):
            return np.sum((v1 - v2) ** 2)

        centers = np.zeros((2, self.num_dimensions))  # Initialize two centroids with zeros
        counters = [0] * 2  # Initialize counters to keep track of the number of points assigned to each centroid

        indices = list(self.indices)  # Create a list from the indices of the current node
        count = 0  # Initialize the update count
        num_steps = max(len(self), MIN_STEPS_K_MEANS)  # Define the minimum number of update steps for k-means

        while True:
            # Shuffle the index list to update centroids in a random order
            random.shuffle(indices)

            for idx in indices:
                # Find the closest centroid to the data point
                distances = [euclidean_distance(data[idx], mu) * c / (1.0 + c) for (mu, c) in zip(centers, counters)]
                min_idx = np.argmin(distances)

                # Update the chosen centroid and its counter based on the data point
                centers[min_idx] = (centers[min_idx] * counters[min_idx] + data[idx]) / (counters[min_idx] + 1)
                counters[min_idx] += 1

                count += 1 # Increment the update count

                if count > num_steps:
                    break

            if count > num_steps:
                break

        # Calculate the direction spanned by the centroids
        split_direction_vector = centers[0] - centers[1]
        # Normalize the direction vector
        split_direction_vector /= np.sqrt(np.sum(split_direction_vector ** 2))

        return split_direction_vector

    # Generates some number m of random directions w by sampling from the d-dimensional unit sphere, and
    # picks the w which maximizes the diameter of projected data from the current node
    def random_projection(self, data):

        num_samples = SAMPLES_FOR_RANDOM_PROJECTION  # Number of random directions to sample

        # Sample directions from a d-dimensional normal distribution
        split_direction_vector = np.random.randn(num_samples, self.num_dimensions)

        # Normalize each sample to get a sample from the unit sphere
        for i in range(num_samples):
            split_direction_vector[i] /= np.sqrt(np.sum(split_direction_vector[i] ** 2))

        # Find the direction that maximally spreads the data:

        # Initialize arrays to store the minimum and maximum similarity values for each direction
        min_val = np.inf * np.ones(num_samples)
        max_val = -np.inf * np.ones(num_samples)

        for i in self.indices:
            # Compute the similarity between each direction and the data point 'i'
            split_direction_similarity = np.dot(split_direction_vector, data[i])
            min_val = np.minimum(min_val, split_direction_similarity)
            max_val = np.maximum(max_val, split_direction_similarity)

        # Select the direction that maximizes the spread (difference between max and min similarity values)
        best_direction_index = np.argmax(max_val - min_val)
        split_direction_vector = split_direction_vector[best_direction_index]

        return split_direction_vector

class SpatialTree_Test_Cases(unittest.TestCase):

    def test_dict(self):
        num_data_points = 5000
        num_dimensions = 20

        # A random projection matrix, for funsies
        random_projection_matrix = np.random.randn(num_dimensions, num_dimensions)

        # For convenience, let's define a random point generator function
        def generate_new_point():
            return np.dot(np.random.randn(num_dimensions), random_projection_matrix)

        print('Generating Data')

        # Now, let's populate the dictionary with N random points
        data = {}

        for i in range(num_data_points):
            # Let's use string-valued keys
            data[f'{i:04x}'] = generate_new_point()

        # Let's make a few distinguished points
        data['Alice'] = generate_new_point()
        data['Bob'] = generate_new_point()
        data['Carol'] = generate_new_point()

        # Construct a tree. Let's use a 2-means tree with a spill percentage of 0.3
        print('Building Tree')
        spatial_tree = SpatialTree(data, split_rule=SplitRule.K_MEANS, spill_fraction=0.3)

        # Show some stats about the constructed tree
        print(f'Number of items in tree: {len(spatial_tree)}')
        print(f'Dimensionality: {spatial_tree.num_dimensions}')
        print(f'Height of tree: {spatial_tree.height}')
        print(f'Spill percentage: {spatial_tree.spill_fraction}')
        print(f'Split rule: {spatial_tree.split_rule}')

        # Let's find the nearest neighbors of 'Bob':
        knn_bob = spatial_tree.k_nearest_neighbors(data, k=10, index='Bob')
        print(f'KNN(Bob): {knn_bob}')

        # Find the nearest neighbors of a random vector:
        knn_random = spatial_tree.k_nearest_neighbors(data, k=10, vector=generate_new_point())
        print(f'KNN(random): {knn_random}')

        # With dictionary-type data, we can add new points to the tree as well
        data['Dave'] = generate_new_point()
        spatial_tree.update({'Dave': data['Dave']})

        # For retrieval purposes, the new point will be included in the tree from then onward
        knn_dave = spatial_tree.k_nearest_neighbors(data, k=10, index='Dave')
        print(f'KNN(Dave): {knn_dave}')

    def test_matrix(self):
        # Generate random data with N data points and D dimensions
        num_data_points = 5000
        num_dimensions = 20
        data = np.random.randn(num_data_points, num_dimensions)

        # Apply a random projection to the data to make it less trivial
        random_projection_matrix  = np.random.randn(num_dimensions, num_dimensions)
        data = np.dot(data, random_projection_matrix )

        # Build a spatial tree with default settings (KD-spill-tree with automatic height and 25% spill fraction)
        print('Building Tree')
        spatial_tree = SpatialTree(data)

        # Show some stats about the constructed tree
        print(f'Number of items in tree: {len(spatial_tree)}')
        print(f'Dimensionality: {spatial_tree.num_dimensions}')
        print(f'Height of tree: {spatial_tree.height}')
        print(f'Spill percentage: {spatial_tree.spill_fraction}')
        print(f'Split rule: {spatial_tree.split_rule}')

        # Create a height=0 tree for comparison against brute-force search
        tree_root = SpatialTree(data, height=0)

        # Find the 10 approximate nearest neighbors of the 500th data point
        knn_a = spatial_tree.k_nearest_neighbors(data, k=10, index=499)
        print(f'KNN approx (index): {knn_a}')

        # Find the true nearest neighbors using the height=0 tree
        knn_t = tree_root.k_nearest_neighbors(data, k=10, index=499)
        print(f'KNN true (index): {knn_t}')

        # Calculate recall rate (proportion of true nearest neighbors found among approximate nearest neighbors)
        recall = len(set(knn_a) & set(knn_t)) / len(set(knn_t))
        print(f'Recall: {recall}')

        # Ensure that our recall rate is greater than 50%
        assert recall >= 0.5

        # Search with a new vector not already in the tree

        # Generate a random test query
        query = np.dot(np.random.randn(num_dimensions), random_projection_matrix )

        # Find approximate nearest neighbors for the test query
        knn_a = spatial_tree.k_nearest_neighbors(data, k=10, vector=query)
        print(f'KNN approx (vector): {knn_a}')

        # Find the true nearest neighbors for the test query using the height=0 tree
        knn_t = tree_root.k_nearest_neighbors(data, k=10, vector=query)
        print(f'KNN true (vector): {knn_t}')

        # Calculate recall rate for the test query
        recall = len(set(knn_a) & set(knn_t)) / len(set(knn_t))
        print(f'Recall: {recall}')

        # Ensure that our recall rate is greater than 50%
        assert recall >= 0.5

if __name__ == "__main__":
    unittest.main()