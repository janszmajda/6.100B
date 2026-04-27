# This tester will ONLY test the functions listed below (lines 13-19)! For the ones
# that generate graphs, you'll have to check it 
# 
# Files expected in the same folder:
#   ps5.py  (student file)
#   titanic.csv 
# -----------------------------------------------------------------------------

import io
import unittest
import contextlib
import random
from collections import Counter

from ps5 import (
    get_titanic_data,
    get_accuracy,
    get_precision,
    get_recall,
    get_f1_score,
    split_data,
    find_best_model,
    evaluate_model,
    generate_synthetic_data,
)

# load data
X, Y = get_titanic_data('titanic_train.csv')
val_X, val_Y = get_titanic_data('titanic_student_tester.csv')
tree_depths_list = [2,4,8,16,32,64]
num_trees_list = [1,5,9,15,25,35,45,95]
num_trials = 5
train_frac = 0.8

def first_bit_fn(in_vec):
    return in_vec[0]

def second_bit_fn(in_vec):
    return in_vec[1]

def avg_bit_half_fn(in_vec):
    return 1 if sum(in_vec) >= len(in_vec)/2 else 0

def count_fn(in_vec):
    return int(in_vec.count(1) >= 2)

###############################################################################
# Part 1 - Model Metrics (Accuracy, Precision, Recall, F1 Score)
###############################################################################

class TestPS5_Part1(unittest.TestCase):
    def test_1_split_data(self):
        result = split_data(X, Y, 0.7)
        
        self.assertEqual(len(result), 4, "split_data must return exactly four objects: train_X, train_Y, test_X, test_Y")
        for result_list in result:
            self.assertIsInstance(result_list, list, "split_data should return four lists")

        self.assertAlmostEqual(len(result[0]), round(len(X) * 0.7), "trainX should be comprised of train_frac of the data")
        self.assertAlmostEqual(len(result[1]), round(len(Y) * 0.7), "trainY should be comprised of train_frac of the data")

        joint_X = result[0] + result[2]
        joint_Y = result[1] + result[3]

        self.assertEqual(len(joint_X), len(X), "there should be no loss of data when splitting X")
        self.assertEqual(len(joint_Y), len(Y), "there should be no loss of data when splitting Y")

        sorted_X = sorted(tuple(row) for row in X)
        sorted_joint_X = sorted(tuple(row) for row in joint_X)

        self.assertEqual(sorted_X, sorted_joint_X, "trainX and testX together should have all the same elements as X")
        self.assertEqual(Counter(joint_Y), Counter(Y), "trainY and testY together should have all the same elements as Y")

        second_run_result = split_data(X, Y, 0.7)

        for dataset in range(len(result)):
            try:
                sorted_first = sorted(tuple(row) for row in result[dataset])
                sorted_second = sorted(tuple(row) for row in second_run_result[dataset])
            except:
                sorted_first = sorted(result[dataset])
                sorted_second = sorted(second_run_result[dataset])

            self.assertTrue(any(
                sorted_first[i] != sorted_second[i]
                for i in range(len(sorted_first))), "split_data should not be deterministic")
            
            
    def test_1_accuracy(self):
        tp, fp, tn, fn = 1, 2, 3, 4
        expected = 0.4
        result = get_accuracy(tp, fp, tn, fn)

        self.assertIsInstance(result, (float, int), "accuracy must be a number.")
        self.assertAlmostEqual(expected, result, places=4)
    
    def test_1_precision(self):
        tp, fp, tn, fn = 1, 2, 3, 4
        expected = 0.3333333333333333
        result = get_precision(tp, fp, tn, fn)

        self.assertIsInstance(result, (float, int), "precision must be a number.")
        self.assertAlmostEqual(expected, result, places=4)
    
    # TODO: make sure this passes even if precision is an int
    def test_1_precision_zero(self):
        tp, fp, tn, fn = 0, 0, 3, 4
        expected = 0
        result = get_precision(tp, fp, tn, fn)

        self.assertIsInstance(result, (float, int), "precision must be a number.")
        self.assertAlmostEqual(expected, result, places=4)

    def test_1_recall(self):
        tp, fp, tn, fn = 1, 2, 3, 4
        expected = 0.2
        result = get_recall(tp, fp, tn, fn)

        self.assertIsInstance(result, (float, int), "recall must be a number.")
        self.assertAlmostEqual(expected, result, places=4)
    
    # TODO: make sure this passes even if accuracy is an int
    def test_1_recall_zero(self):
        tp, fp, tn, fn = 0, 2, 3, 0
        expected = 0
        result = get_recall(tp, fp, tn, fn)

        self.assertIsInstance(result, (float, int), "recall must be a number.")
        self.assertAlmostEqual(expected, result, places=4)

    def test_1_f1_score(self):
        tp, fp, tn, fn = 1, 2, 3, 4
        expected = 0.25
        result = get_f1_score(tp, fp, tn, fn)

        self.assertIsInstance(result, (float, int), "f1 score must be a number.")
        self.assertAlmostEqual(expected, result, places=4)
    
    def test_1_f1_score_zero(self):
        tp, fp, tn, fn = 0, 0, 0, 0
        expected = 0
        result = get_f1_score(tp, fp, tn, fn)

        self.assertIsInstance(result, (float, int), "f1 score must be a number.")
        self.assertAlmostEqual(expected, result, places=4)


###############################################################################
# Part 2 - Titanic Analysis (Find best model)
###############################################################################

class TestPS5_Part2(unittest.TestCase):

    def test_2_find_best_model_accuracy(self):
        tree_depth, num_trees, model = find_best_model(X, Y, get_accuracy, tree_depths_list, num_trees_list, num_trials, train_frac)
        expected_lower = 0.80898 # 0.5th percentile
        expected_upper = 0.842505 # 99.5th percentile
        
        result_acc = evaluate_model(val_X, val_Y, get_accuracy, model)
        self.assertIsInstance(result_acc, (float, int), "accuracy must be a float.")
        self.assertLessEqual(expected_lower, result_acc, f"you had an accuracy of {result_acc}, which was lower than the 0.5th percentile of our trials, 0.80898. Check your find_best_model and evaluate_model functions.")
        self.assertLessEqual(result_acc, expected_upper, f"you had an accuracy of {result_acc}, which was higher than the 99.5th percentile of our trials, 0.842505. Check your find_best_model and evaluate_model functions.")
    

    def test_2_find_best_model_precision(self):
        tree_depth, num_trees, model = find_best_model(X, Y, get_precision, tree_depths_list, num_trees_list, num_trials, train_frac)
        expected_lower = 0.7578732902735562 # 0.5th percentile
        expected_upper = 0.9411513042870131 # 99.5th percentile
        
        result_precision = evaluate_model(val_X, val_Y, get_precision, model)
        self.assertIsInstance(result_precision, (float, int), "precision must be a float.")
        self.assertLessEqual(expected_lower, result_precision, f"you had a precision of {result_precision}, which was lower than the 0.5th percentile of our trials, 0.7578732902735562. Check your find_best_model and evaluate_model functions.")
        self.assertLessEqual(result_precision, expected_upper, f"you had a precision of {result_precision}, which was higher than the 99.5th percentile of our trials, 0.9411513042870131. Check your find_best_model and evaluate_model functions.")


    def test_2_find_best_model_f1(self):
        tree_depth, num_trees, model = find_best_model(X, Y, get_f1_score, tree_depths_list, num_trees_list, num_trials, train_frac)
        expected_lower = 0.7278338735629504 # 0.5th percentile
        expected_upper = 0.7699077983002446 # 99.5th percentile
        
        result_f1 = evaluate_model(val_X, val_Y, get_f1_score, model)
        self.assertIsInstance(result_f1, (float, int), "f1 must be a float.")
        self.assertLessEqual(expected_lower, result_f1, f"you had an f1 score of {result_f1}, which was lower than the 0.5th percentile of our trials, 0.7278338735629504. Check your find_best_model and evaluate_model functions.")
        self.assertLessEqual(result_f1, expected_upper, f"you had an f1 score of {result_f1}, which was higher than the 99.5th percentile of our trials, 0.7699077983002446. Check your find_best_model and evaluate_model functions.")


###############################################################################
# Part 3 - Understanding the Model (Only autograding generate_synthetic_data)
###############################################################################

class TestPS5_Part3(unittest.TestCase):

    def test_3_gen_synthetic_data_label(self): # only noising the output
        n_flips_sum = 0
        n_trials = 100

        for _ in range(n_trials):
            X, Y = generate_synthetic_data(100, 4, first_bit_fn, feature_noise_std=0, label_flip_prob=0.3)

            cur_n_flips = 0
            for x, syn_y in zip(X, Y):
                actual_y = first_bit_fn(x)
                if actual_y != syn_y:
                    cur_n_flips += 1
            
            n_flips_sum += cur_n_flips
        
        avg_n_flips = n_flips_sum / n_trials

        expected_lower = 28.81 # 0.5th percentile of # of flipped bits in data
        expected_upper = 31.18 # 99.5th percentile of # of flipped bits in data
        
        self.assertLessEqual(expected_lower, avg_n_flips, f"we found {avg_n_flips} flips on average across 100 runs of your generate_synthetic_data function, which was lower than the 0.5th percentile of our trials, 28.81.")
        self.assertLessEqual(avg_n_flips, expected_upper, f"we found {avg_n_flips} flips on average across 100 runs of your generate_synthetic_data function, which was higher than the 99.5th percentile of our trials, 31.18.")


    def test_3_gen_synthetic_data_input(self): # only noising the input        
        n_flips_sum = 0
        n_trials = 100

        num_features = 4
        num_examples = 100

        feature_noise_std = 0.7

        expected_mean = 0.5
        expected_var = 0.25 + feature_noise_std**2

        sample_means = {}
        sample_vars = {}

        for _ in range(n_trials):
            X, Y = generate_synthetic_data(num_examples, num_features, first_bit_fn, feature_noise_std, label_flip_prob=0)
            for feature_index in range(num_features):
                all_examples_for_feature = [X[exam_num][feature_index] for exam_num in range(num_examples)]
                sample_mean = sum(all_examples_for_feature) / len(all_examples_for_feature)
                try:
                    sample_means[feature_index].append(sample_mean)
                except:
                    sample_means[feature_index] = [sample_mean]
                sample_var = sum([(x - sample_mean)**2 for x in all_examples_for_feature]) / (len(all_examples_for_feature) - 1)
                try:
                    sample_vars[feature_index].append(sample_var)
                except:
                    sample_vars[feature_index] = [sample_var]

                mean_se = (sample_var ** 0.5) / (num_features ** 0.5)
                self.assertAlmostEqual(sample_mean, expected_mean, delta=3 * mean_se, msg=f"Feature {feature_index} mean {sample_mean:.3f} too far from {expected_mean}")

            
        for feature_index in range(num_features):
            avg_var = sum(sample_vars[feature_index]) / n_trials
            var_se = expected_var * (2 / (num_examples - 1)) ** 0.5
            mean_of_var_se = var_se / n_trials ** 0.5
            self.assertAlmostEqual(avg_var, expected_var, delta=3 * mean_of_var_se, msg=f"Feature {feature_index} variance {avg_var:.3f} too far from {expected_var}")



    def test_3_gen_synthetic_data_label_count_fn(self): # only noising output, and using count function instead of first bit
        n_flips_sum = 0
        n_trials = 100

        for _ in range(n_trials):
            X, Y = generate_synthetic_data(100, 4, count_fn, feature_noise_std=0, label_flip_prob=0.3)

            cur_n_flips = 0
            for x, syn_y in zip(X, Y):
                actual_y = count_fn(x)
                if actual_y != syn_y:
                    cur_n_flips += 1
            
            n_flips_sum += cur_n_flips
        
        avg_n_flips = n_flips_sum / n_trials

        expected_lower = 28.839949999999998 # 0.5th percentile of # of flipped bits in data
        expected_upper = 31.170049999999993 # 99.5th percentile of # of flipped bits in data
        
        self.assertLessEqual(expected_lower, avg_n_flips, f"we found {avg_n_flips} flips on average across 100 runs of your generate_synthetic_data function, which was lower than the 0.5th percentile of our trials, 28.839949999999998.")
        self.assertLessEqual(avg_n_flips, expected_upper, f"we found {avg_n_flips} flips on average across 100 runs of your generate_synthetic_data function, which was higher than the 99.5th percentile of our trials, 31.170049999999993.")


###############################################################################
# Optional grading metadata
###############################################################################

point_values = {
    "test_1_accuracy": 0.1,
    "test_1_precision": 0.1,
    "test_1_precision_zero": 0.1,
    "test_1_recall": 0.1,
    "test_1_recall_zero": 0.1,
    "test_1_f1_score": 0.1,
    "test_1_f1_score_zero": 0.1,
    "test_2_find_best_model_accuracy": 0.5,
    "test_2_find_best_model_precision": 0.5,
    "test_2_find_best_model_f1": 0.5,
    "test_3_gen_synthetic_data_label": 0.3,
    "test_3_gen_synthetic_data_input": 0.3,
    "test_3_gen_synthetic_data_label_count_fn": 0.3,
}

error_messages = {
    "test_1_accuracy": "Your code for Part 1 (model metrics) produced an error.",
    "test_1_precision": "Your code for Part 1 (model metrics) produced an error.",
    "test_1_precision_zero": "Your code for Part 1 (model metrics) produced an error.",
    "test_1_recall": "Your code for Part 1 (model metrics) produced an error.",
    "test_1_recall_zero": "Your code for Part 1 (model metrics) produced an error.",
    "test_1_f1_score": "Your code for Part 1 (model metrics) produced an error.",
    "test_1_f1_score_zero": "Your code for Part 1 (model metrics) produced an error.",
    "test_2_find_best_model_accuracy": "Your code for Part 2 (find best model, evaluate models) produced an error.",
    "test_2_find_best_model_precision": "Your code for Part 2 (find best model, evaluate models) produced an error.",
    "test_2_find_best_model_f1": "Your code for Part 2 (find best model, evaluate models) produced an error.",
    "test_3_gen_synthetic_data_label": "Your code for part 3 (generate synthetic data) produced an error.",
    "test_3_gen_synthetic_data_input": "Your code for part 3 (generate synthetic data) produced an error.",
    "test_3_gen_synthetic_data_label_count_fn": "Your code for part 3 (generate synthetic data) produced an error.",
}

failure_messages = {
    "test_1_accuracy": "Your code for Part 1 (model metrics) failed the test case.",
    "test_1_precision": "Your code for Part 1 (model metrics) failed the test case.",
    "test_1_precision_zero": "Your code for Part 1 (model metrics) failed the test case.",
    "test_1_recall": "Your code for Part 1 (model metrics) failed the test case.",
    "test_1_recall_zero": "Your code for Part 1 (model metrics) failed the test case.",
    "test_1_f1_score": "Your code for Part 1 (model metrics) failed the test case.",
    "test_1_f1_score_zero": "Your code for Part 1 (model metrics) failed the test case.",
    "test_2_find_best_model_accuracy": "Your code for Part 2 (find best model, evaluate models) failed the test case.",
    "test_2_find_best_model_precision": "Your code for Part 2 (find best model, evaluate models) failed the test case.",
    "test_2_find_best_model_f1": "Your code for Part 2 (find best model, evaluate models) failed the test case.",
    "test_3_gen_synthetic_data_label": "Your code for part 3 (generate synthetic data) failed the test case.",
    "test_3_gen_synthetic_data_input": "Your code for part 3 (generate synthetic data) failed the test case.",
    "test_3_gen_synthetic_data_label_count_fn": "Your code for part 3 (generate synthetic data) failed the test case.",
}

###############################################################################
# Runner
###############################################################################

if __name__ == "__main__":
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestPS5_Part1))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestPS5_Part2))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestPS5_Part3))
    all_suites = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=2).run(all_suites)
