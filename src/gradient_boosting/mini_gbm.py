import sys
import time
import unittest

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets

import numpy as np
from collections import namedtuple

# GBT Parameters:
Parameters = namedtuple("Parameters", "GAMMA "
                                      "LAMBDA "
                                      "MIN_SPLIT_GAIN "
                                      "MAX_DEPTH "
                                      "INITIAL_SHRINKAGE_RATE "
                                      "LEARNING_RATE "
                                      "NUM_BOOST_ROUNDS "
                                      "EARLY_STOPPING_ROUNDS")

parameters = Parameters(   GAMMA = 0.
                         , LAMBDA = 1.
                         , MIN_SPLIT_GAIN=0.1
                         , MAX_DEPTH = 6
                         , INITIAL_SHRINKAGE_RATE = 1.
                         , LEARNING_RATE = 0.3
                         , NUM_BOOST_ROUNDS = 20
                         , EARLY_STOPPING_ROUNDS = 5 )

DataSet = namedtuple("DataSet", "X y")

class TreeNode(object):

    def __init__(self):
        global parameters
        self.parameters = parameters
        self.is_leaf = False
        self.left_child = None
        self.right_child = None

        self.split_feature_id = None
        self.split_value = None
        self.weight = None

    def loss_scoring_function(self, gradient, hessian):
       return np.square(gradient) / (hessian + self.parameters.LAMBDA)

    def calc_loss_reduction(self, current_gradient, current_hessian
                            , left_gradient, left_hessian
                            , right_gradient, right_hessian):

        return self.loss_scoring_function(left_gradient, left_hessian) \
               + self.loss_scoring_function(right_gradient, right_hessian) \
               - self.loss_scoring_function(current_gradient, current_hessian)

    def calc_leaf_weight(self, gradient, hessian):
        return np.sum(gradient) / (np.sum(hessian) + self.parameters.LAMBDA)

    def mutate_into_leaf(self, gradient, hessian, shrinkage_rate):
        self.weight = self.calc_leaf_weight(gradient, hessian) * shrinkage_rate
        self.is_leaf = True

    def build_tree(self, data, gradient, hessian, shrinkage_rate, depth):
        num_rows = data.shape[0]
        assert num_rows == len(gradient) == len(hessian)

        if depth <= self.parameters.MAX_DEPTH:

            best_gain = 0.
            best_feature_id = None
            best_value = 0.
            best_left_tree_indices = None
            best_right_tree_indices = None

            gradient_total = np.sum(gradient)
            hessian_total = np.sum(hessian)

            num_features = data.shape[1]

            for feature_id in range(num_features):
                left_gradient = 0.
                left_hessian = 0.
                # Perform an indirect sort along the feature axis, returning an array of indices that index the
                # data array in sorted order (sorted by feature_id):
                sorted_features = data[:, feature_id].argsort()
                for current_row_index in range(num_rows):

                    left_gradient = left_gradient + gradient[sorted_features[current_row_index]]
                    right_gradient = gradient_total - left_gradient
                    left_hessian = left_hessian + hessian[sorted_features[current_row_index]]
                    right_hessian = hessian_total - left_hessian

                    current_gain = self.calc_loss_reduction(  current_gradient=gradient_total
                                                            , current_hessian=hessian_total
                                                            , left_gradient=left_gradient
                                                            , left_hessian=left_hessian
                                                            , right_gradient=right_gradient
                                                            , right_hessian=right_hessian )

                    if current_gain > best_gain:
                        best_gain = current_gain
                        best_feature_id = feature_id
                        best_value = data[sorted_features[current_row_index]][feature_id]
                        best_left_tree_indices = sorted_features[:current_row_index + 1]
                        best_right_tree_indices = sorted_features[current_row_index + 1:]

            if best_gain >= self.parameters.MIN_SPLIT_GAIN:
                self.split_feature_id = best_feature_id
                self.split_value = best_value

                self.left_child = TreeNode()
                self.left_child.build_tree( data = data[best_left_tree_indices]
                                            , gradient = gradient[best_left_tree_indices]
                                            , hessian = hessian[best_left_tree_indices]
                                            , shrinkage_rate=shrinkage_rate
                                            , depth = depth + 1)

                self.right_child = TreeNode()
                self.right_child.build_tree(data=data[best_right_tree_indices]
                                           , gradient=gradient[best_right_tree_indices]
                                           , hessian=hessian[best_right_tree_indices]
                                           , shrinkage_rate=shrinkage_rate
                                           , depth=depth + 1)

            else:
                self.mutate_into_leaf(gradient, hessian, shrinkage_rate)

        else:
            self.mutate_into_leaf(gradient, hessian, shrinkage_rate)

    def predict(self, data):
        if self.is_leaf:
            return self.weight
        else:
            if data[self.split_feature_id] <= self.split_value:
                return self.left_child.predict(data)
            else:
                return self.right_child.predict(data)



class Tree(object):

    def __init__(self):
        self.root = None

    def build_tree(self, data, gradient, hessian, shrinkage_rate):
        assert len(data) == len(gradient) == len(hessian)
        self.root = TreeNode()
        self.root.build_tree(data, gradient, hessian, shrinkage_rate, depth = 0)

    def predict(self, data):
        return self.root.predict(data)

    def __str__(self):

        def construct_branching_rules(node, depth):

            branching_logic_str = ''
            indent = "  " * depth
            if not node.is_leaf:
                branching_logic_str += f"{indent}if feature with id {node.split_feature_id} <= {node.split_value}:\n"
                branching_logic_str += construct_branching_rules(node.left_child, depth + 1)
                branching_logic_str += f"{indent}else:  # if feature with id {node.split_feature_id} > {node.split_value}\n"
                branching_logic_str += construct_branching_rules(node.right_child, depth + 1)
            else:
                branching_logic_str += f"{indent}return {node.weight}\n"

            return branching_logic_str

        return construct_branching_rules(self.root, 1)

class Gradient_Boosted_Tree(object):

    def __init__(self):
        global parameters
        self.parameters = parameters
        self.best_iteration = None

    def calc_training_data_scores(self, training_set, models):
        if len(models) == 0:
            return None

        training_data = training_set.X
        scores = np.zeros(len(training_data))

        for i in range(len(training_data)):
            scores[i] = self.predict(training_data[i], models)

        return scores

    def calc_l2_gradient(self, training_set, scores):
        labels = training_set.y
        num_rows = len(labels)
        hessian = np.full(shape=num_rows, fill_value=2)

        if scores is not None:
            gradient = np.array([2 * (labels[i] - scores[i]) for i in range(num_rows)])
        else:
            # draw samples uniformly between [0, 1)
            gradient = np.random.uniform(size=num_rows)

        return gradient, hessian

    def calc_derivative_vectors(self, training_set, scores):
        return self.calc_l2_gradient(training_set, scores)

    def calc_l2_loss(self, data_set, models):
        errors = []

        for X, y in zip(data_set.X, data_set.y):
            errors.append(y - self.predict(X, models))

        return np.mean(np.square(errors))

    def calc_loss(self, data_set, models):
        return self.calc_l2_loss(data_set, models)

    def build_learner(self, training_set, gradient, hessian, shrinkage_rate):
        learner = Tree()
        learner.build_tree(training_set.X, gradient, hessian, shrinkage_rate)
        return learner

    def train(self, training_set, validation_set = None, output_training_time = False):
        models = []
        shrinkage_rate = self.parameters.INITIAL_SHRINKAGE_RATE
        best_iteration = None
        best_validation_loss_value = sys.maxsize

        if output_training_time:
            training_start_time = time.time()

        for i in range(self.parameters.NUM_BOOST_ROUNDS):

            scores = self.calc_training_data_scores(training_set, models)
            gradient, hessian = self.calc_derivative_vectors(training_set, scores)
            learner = self.build_learner(training_set, gradient, hessian, shrinkage_rate)

            if i > 0:
                shrinkage_rate = shrinkage_rate * self.parameters.LEARNING_RATE

            models.append(learner)

            validation_loss = self.calc_loss(validation_set, models) if validation_set else None

            if validation_loss is not None and validation_loss < best_validation_loss_value:
                best_validation_loss_value = validation_loss
                best_iteration = i

            if (i - best_iteration >= self.parameters.EARLY_STOPPING_ROUNDS):
                break

        self.models = models
        self.best_iteration = best_iteration

        if output_training_time:
            print("Training finished. The total time elapsed is: {:.2f} secs".format(time.time() - training_start_time))

    def predict(self, data, models=None):
        if models is None:
            models = self.models
        assert models is not None
        predictions_array = np.fromiter((model.predict(data) for model in models), float)
        return np.sum(predictions_array)

    def __str__(self):
        gbt_string = ''
        for index, model in enumerate(self.models):
            gbt_string += f"Tree {index}:\n"
            gbt_string += str(model)
        return gbt_string

class Tiny_GBT_Test_Cases(unittest.TestCase):

    def setUp(self):
        self.VALIDATION_PERCENTAGE = 0.25
        self.RANDOM_STATE = 24
        self.OUTPUT_VALIDATION_RESULTS = True
        self.OUTPUT_TRAINING_TIME = True
        self.OUTPUT_TREE_STRUCTURE = False

    def test_iris_data(self):

        data_set = datasets.load_iris()
        data, labels = data_set.data, data_set.target

        training_data, validation_data, training_labels, validation_labels = train_test_split(  data
                                                                                              , labels
                                                                                              , test_size = self.VALIDATION_PERCENTAGE
                                                                                              , random_state = self.RANDOM_STATE )

        training_data_set = DataSet(X=training_data
                                    , y=training_labels)

        validation_data_set = DataSet(X=validation_data
                                      , y=validation_labels)

        gradient_boosted_tree = Gradient_Boosted_Tree()
        gradient_boosted_tree.train(  training_data_set
                                    , validation_data_set
                                    , output_training_time = self.OUTPUT_TRAINING_TIME )

        if self.OUTPUT_TREE_STRUCTURE:
            print("Outputting iris data set GBT structure: ")
            print(str(gradient_boosted_tree))

        predictions = []

        for x in validation_data:
            predictions.append(
                gradient_boosted_tree.predict(x)
            )

        validation_targets = [abs(round(float(value), 0)) for value in validation_labels]
        validation_predictions = [abs(round(float(value), 0)) for value in predictions]

        if self.OUTPUT_VALIDATION_RESULTS:
            print("Outputting iris validation data results: ")
            print(f"Validation set labels : {validation_targets}")
            print(f"Validation predictions: {validation_predictions}")

        validation_rmse = mean_squared_error(validation_labels, predictions) ** 0.5

        print(f"Validation RMSE: {validation_rmse}")

        self.assertTrue(validation_rmse < 0.5)

    def test_breast_cancer_data(self):

        data_set = datasets.load_breast_cancer()
        data, labels = data_set.data, data_set.target

        training_data, validation_data, training_labels, validation_labels = train_test_split(  data
                                                                                              , labels
                                                                                              , test_size = self.VALIDATION_PERCENTAGE
                                                                                              , random_state = self.RANDOM_STATE )

        training_data_set = DataSet(X=training_data
                                    , y=training_labels)

        validation_data_set = DataSet(X=validation_data
                                      , y=validation_labels)

        gradient_boosted_tree = Gradient_Boosted_Tree()
        gradient_boosted_tree.train(  training_data_set
                                    , validation_data_set
                                    , output_training_time = self.OUTPUT_TRAINING_TIME )

        if self.OUTPUT_TREE_STRUCTURE:
            print("Outputting breast cancer data set GBT structure: ")
            print(str(gradient_boosted_tree))

        predictions = []

        for x in validation_data:
            predictions.append(
                gradient_boosted_tree.predict(x)
            )

        validation_targets = [abs(round(float(value), 0)) for value in validation_labels]
        validation_predictions = [abs(round(float(value), 0)) for value in predictions]

        if self.OUTPUT_VALIDATION_RESULTS:
            print("Outputting breast cancer validation data results: ")
            print(f"Validation set labels : {validation_targets}")
            print(f"Validation predictions: {validation_predictions}")

        validation_rmse = mean_squared_error(validation_labels, predictions) ** 0.5

        print(f"Validation RMSE: {validation_rmse}")

        self.assertTrue(validation_rmse < 0.5)

    def test_boston_data(self):

        data_set = datasets.load_wine()
        data, labels = data_set.data, data_set.target

        training_data, validation_data, training_labels, validation_labels = train_test_split(  data
                                                                                              , labels
                                                                                              , test_size = self.VALIDATION_PERCENTAGE
                                                                                              , random_state = self.RANDOM_STATE )

        training_data_set = DataSet(X=training_data
                                    , y=training_labels)

        validation_data_set = DataSet(X=validation_data
                                      , y=validation_labels)

        gradient_boosted_tree = Gradient_Boosted_Tree()
        gradient_boosted_tree.train(  training_data_set
                                    , validation_data_set
                                    , output_training_time = self.OUTPUT_TRAINING_TIME )

        if self.OUTPUT_TREE_STRUCTURE:
            print("Outputting boston data set GBT structure: ")
            print(str(gradient_boosted_tree))

        predictions = []

        for x in validation_data:
            predictions.append(
                gradient_boosted_tree.predict(x)
            )

        validation_targets = [abs(round(float(value), 0)) for value in validation_labels]
        validation_predictions = [abs(round(float(value), 0)) for value in predictions]

        if self.OUTPUT_VALIDATION_RESULTS:
            print("Outputting boston validation data results: ")
            print(f"Validation set labels : {validation_targets}")
            print(f"Validation predictions: {validation_predictions}")

        validation_rmse = mean_squared_error(validation_labels, predictions) ** 0.5

        print(f"Validation RMSE: {validation_rmse}")

        self.assertTrue(validation_rmse < 8)

if __name__ == "__main__":
    unittest.main()