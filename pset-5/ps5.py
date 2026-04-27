import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

### CODE PROVIDED FOR YOUR USE: DO NOT MODIFY ###
def plot_conf_matrix(tp, fp, tn, fn, title, caption = None):
    """
    Plot confusion matrix with the given true positive, false positive, true negative, and
    false negative values.
    Params:
        tp, fp, tn, fn: positive integers representing true positive, false positive, true negative,
            and false negative values respectively.
        title, caption: title and optional caption for confusion matrix.
    Returns: None, but displays confusion matrix.
    """
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
    plt.show()


def get_titanic_data(filename):
    """
    Reads titanic data from CSV, and returns X, Y where X is a 2D numpy array (which can
    be treated as a nested list where each element is a list representing all information
    about the passenger except whether they survived), and Y a 1D numpy array indicating for
    each person, whether they survived (as 0 or 1).
    """
    df = pd.read_csv(filename)
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Split into features and label
    X = df.drop('Survived', axis=1)
    Y = df['Survived']
    # print(X)
    # print(Y)
    return X.to_numpy(), Y.to_numpy()


####### PART 1: GROUNDWORK #######

def split_data(X, Y, train_frac):
    """
    *Randomly* splits data into two disjoint partitions.
    (e.g. training set and validation set, or training set and test set)
    Note: this function cannot be deterministic.

    Params:
        X (list): datapoints for the input (data can be multidimensional)
        Y (list): labels (for a binary classifier, each output is 0 or 1)
            X and Y must be the same length.
        train_frac (float): fraction of the data that should be used for the training set,
            and the remaining data will be the validation/test set.

    Returns (tuple of lists): train_X, train_Y, test_X, test_Y
        train_X and train_Y comprise the features and labels (respectively) used for the
        training set, while test_X and test_Y make up the features and labels in the
        test/validation set. train_X and test_X together should have all of the same data
        as X (no data loss, no extra data). The same applies for train_Y and test_Y with
        respect to Y. test_X and test_Y must be the same length. train_X and train_Y must
        be the same length = 0.7 * len(X).
    """
    # Combine lists using zip and randomly shuffle
    data = list(zip(X, Y))
    random.shuffle(data)

    # Making training data and test data lists
    train_len = round(len(data) * train_frac)
    train_data = data[:train_len]
    test_data = data[train_len:]

    # Using zip to pick out shich elements we want in lists
    train_X = [x for x, y in train_data]
    train_Y = [y for x, y in train_data]
    test_X = [x for x, y in test_data]
    test_Y = [y for x, y in test_data]

    return train_X, train_Y, test_X, test_Y


def get_accuracy(tp, fp, tn, fn):
    if tp + fp + tn + fn == 0:
        return 0
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy


def get_precision(tp, fp, tn, fn):
    if tp + fp == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        return precision


def get_recall(tp, fp, tn, fn):
    if tp + fn == 0:
        return 0
    else:
        recall = tp / (tp + fn)
        return recall


def get_f1_score(tp, fp, tn, fn):
    precision = get_precision(tp, fp, tn, fn)
    recall = get_recall(tp, fp, tn, fn)

    if precision + recall == 0:
        return 0
    else:
        f1 = 2 * ((precision) * (recall)) / (precision + recall)
        return f1


###### PART 2: TITANIC ANALYSIS ######
def find_best_model(X, Y, objective_fn, tree_depths_list, num_trees_list, num_trials, train_frac=0.8):
    """
    Trains a RandomForestClassifier model with every combination of hyperparameters
    (tree depth and number of trees), and outputs the model that performs best under the input
    objective function. The final model should be trained on the entire training data set.

    Params:
        X: training data input, used for both training and validation according to train_frac.
        Y: training data labels (output), used for both train and val according to train_frac.
        objective_fn: a FUNCTION that takes the model's (true pos, false pos, true neg, false neg)
                        as input and outputs a numerical metric (float) (e.g. accuracy, precision).
        tree_depths_list: a list of positive integer values for tree depth of the model.
        num_trees_list: a list of positive integer values for # of trees for the model.
        num_trials: # of train/val splits to generate and train/evaluate models on.
        train_frac: fraction of data that should be used as the training set when creating train/val splits.

    Returns: (tree_depth, num_trees, model) for the model that gets the max score under objective_fn.
    """

    model_dict = {}
    # Train model on training data for each tree depth and num trees
    for tree_depth in tree_depths_list:
        for num_trees in num_trees_list:
            grades_list = []
            for _ in range(num_trials):
                # Split data into two partitions: training and validation
                train_X, train_Y, test_X, test_Y = split_data(X, Y, train_frac)

                model = RandomForestClassifier(max_depth=tree_depth, n_estimators=num_trees)
                model.fit(train_X, train_Y)

                # Calculate model performance on the validation data
                score = evaluate_model(test_X, test_Y, objective_fn, model)
                grades_list.append(score)
            grades_average = sum(grades_list) / len(grades_list)
            model_dict[grades_average] = (tree_depth, num_trees, model)
    max_k = 0
    for k in model_dict.keys():
        if k > max_k:
            max_k = k
    best_model_vals = model_dict[max_k]

    # Train and return best model
    best_model = RandomForestClassifier(max_depth=best_model_vals[0], n_estimators=best_model_vals[1])
    best_model.fit(X,Y)
    return best_model_vals[0], best_model_vals[1], best_model

def evaluate_model(val_X, val_Y, objective_fn, model):
    """
    Evaluates the given model according to the provided objective function (accuracy, precision, recall, or f1).

    Params:
        val_X: data input to evaluate performance on
        val_Y: data output (labels) to evaluate performance on
        objective_fn: a FUNCTION that takes the model's (true pos, true neg, false pos, false neg)
                        as input and outputs a numerical metric (float) (e.g. accuracy, precision).
        model: the sklearn RandomForestClassifier model

    Returns: the model's score (as a float) under the given objective fn.
    """
    tp, fp, tn, fn = objective_fn_vals(val_X, val_Y, model)

    score = objective_fn(tp, fp, tn, fn)
    return score

def objective_fn_vals(X, Y, model):
    """
    Gives the tp, fn, fp, and tn values needed for objective function analysis

    Params:
        X: data input to evaluate performance on
        Y: data output to evaluate performance on

    Returns: tuple of the (tp, fp, tn, fn)
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    predictions = model.predict(X) # Returns a list of predictions [0,1,0,...]

    for i in range(len(Y)):
        actual = Y[i]
        predicted = predictions[i]
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 0 and predicted == 0:
            tn += 1

    return tp, fp, tn, fn

# this function does not have any autograder tests.
def plot_model_performance(X, Y, test_X, test_Y, tree_depths_list, num_trees_list, num_trials, train_frac=0.8):
    """
    Uses training set X, Y and given hyperparameter options to train the model that performs the best under
    each of the four metrics, then plots a confusion matrix of that model's performance on the TEST set.

    Params:
        X and Y: lists containing input and output, respectively for training/validation.
        tree_depths_list: a list of positive integer values for tree depth of the model.
        num_trees_list: a list of positive integer values for # of trees for the model.
        num_trials: # of train/val splits to generate and train/evaluate models on.
        train_frac: fraction of data that should be used as the training set when creating train/val splits.

    Returns: None (just displays confusion matrices and statistics for the four metric types)
    """
    metrics = [("Accuracy", get_accuracy), ("Precision", get_precision), ("Recall", get_recall), ("F1 Score", get_f1_score)]

    for metric_name, metric_fn in metrics:
        tree_depth, num_trees, model = find_best_model(X, Y, metric_fn, tree_depths_list, num_trees_list, num_trials, train_frac)

        tp, fp, tn, fn = objective_fn_vals(test_X, test_Y, model)

        # print("Model optimized for " + metric_name + ": ")
        # print("Tree depth: " + str(tree_depth) + " and number of trees: " + str(num_trees))
        # print("Accuracy: " + str(get_accuracy(tp, fp, tn, fn)))
        # print("Precision: " + str(get_precision(tp, fp, tn, fn)))
        # print("Recall: " + str(get_recall(tp, fp, tn, fn)))
        # print("F1 Score: " + str(get_f1_score(tp, fp, tn, fn)))
        # print()

        title = "Confusion Matrix for Model Optimized for " + metric_name
        plot_conf_matrix(tp, fp, tn, fn, title)


def generate_synthetic_data(num_examples, num_features, ground_truth_fn, feature_noise_std, label_flip_prob):
    """
    Generate synthetic classification data, with noise in feature and/or label.
    Note: the function will generate a random binary vector (list of 0s and 1s) for each input,
        find the corresponding ground-truth output with ground_truth_fn AND THEN
        apply Gaussian noise with a mean of 0 and a standard deviation of feature_noise_std
        to each input feature, and flip each output bit with probability label_flip_prob.
        Noise is applied AFTER the correct label is determined.

    Params:
        num_examples: int > 0, total # of datapoints (examples) to generate.
        num_features: int > 0, dimension of feature vector
        ground_truth_fn: function that maps a feature vector (list of 0s
                        and 1s of length num_features) to a single binary output (0 or 1).
        feature_noise_std: float, standard deviation of Gaussian noise for a single input feature
        label_flip_prob: float, probability of flipping an output label bit

    Returns: (X, Y). X is a 2D nested list with len(X) == num_examples and
                    len(X[i]) == num_features for any i. Y is a list of length num_examples.
                    All values are binary (integer 0 or 1).
    """
    X = []
    Y = []

    for _ in range(num_examples):
        # Generate random binary example
        example = []
        for _ in range(num_features):
            example.append(random.randint(0, 1))

        label = ground_truth_fn(example)

        # Add noise to features and label
        noisy_example = []
        for feature in example:
            noisy_feature = feature + random.gauss(0, feature_noise_std)
            noisy_example.append(noisy_feature)

        if random.random() < label_flip_prob:
            label = 1 - label

        X.append(noisy_example)
        Y.append(label)

    return X, Y


def plot_accuracy_over_noise(tree_depth, num_trees, train_set_size, test_set_size, n_input_features, n_trials):
    """
    Trains model on noisy training set and calculates model's accuracy on no-noise test set for
    n_trials trials, then plots average accuracy against the fraction of noise in the training set.
    Displays two plots: one for flips in input bits, and one for flips in output labels. Each of
    three ground truth functions are displayed as a separate line on the graph.
    (Note: for each trial, generates a new synthetic data set)

    Params:
        tree_depth, num_trees: hyperparameters for random forest training
        train_set_size, test_set_size: number of examples of synthetic data to generate for
            training and test sets.
        n_input_features: number of features (dimensions) in each input example
        n_trials: number of trials of synthetic data generation + model training, across which
            accuracy will be averaged.
    """
    def first_feature_fn(example):
        if example[0] < 0.5:
            return 1
        else:
            return 0

    def first_or_second_feature_fn(example):
        if example[0] < 0.5 or example[1] < 0.5:
            return 1
        else:
            return 0

    def average_feature_fn(example):
        if sum(example) / len(example) >= 0.5:
            return 1
        else:
            return 0

    ground_truth_functions = [(first_feature_fn, "First feature less than 0.5"), (first_or_second_feature_fn, "First or second feature is less than 0.5"), (average_feature_fn, "1 if the average feature value is >= 1/2, else 0")]

    noise_values = []
    for i in range(41):
        noise_values.append(i * 0.025)

    # Plot accuracy when features are noisy
    plt.figure()
    for ground_truth_fn, label in ground_truth_functions:
        accuracies = []

        for noise in noise_values:
            total_accuracy = 0

            for _ in range(n_trials):
                train_X, train_Y = generate_synthetic_data(train_set_size, n_input_features, ground_truth_fn, noise, 0)
                test_X, test_Y = generate_synthetic_data(test_set_size, n_input_features, ground_truth_fn, 0, 0)

                model = RandomForestClassifier(max_depth=tree_depth, n_estimators=num_trees)
                model.fit(train_X, train_Y)

                total_accuracy += evaluate_model(test_X, test_Y, get_accuracy, model)

            accuracies.append(total_accuracy / n_trials)

        plt.plot(noise_values, accuracies, marker="o", label=label)

    plt.title("Accuracy vs. standard deviation of Gaussian noise in features data")
    plt.xlabel("Standard deviation of noise in features training data")
    plt.ylabel("Accuracy on non-noised test set")
    plt.legend()
    plt.show()

    # Plot accuracy when labels are noisy
    plt.figure()
    for ground_truth_fn, label in ground_truth_functions:
        accuracies = []

        for noise in noise_values:
            total_accuracy = 0

            for _ in range(n_trials):
                train_X, train_Y = generate_synthetic_data(train_set_size, n_input_features, ground_truth_fn, 0, noise)
                test_X, test_Y = generate_synthetic_data(test_set_size, n_input_features, ground_truth_fn, 0, 0)

                model = RandomForestClassifier(max_depth=tree_depth, n_estimators=num_trees)
                model.fit(train_X, train_Y)

                total_accuracy += evaluate_model(test_X, test_Y, get_accuracy, model)

            accuracies.append(total_accuracy / n_trials)

        plt.plot(noise_values, accuracies, marker="o", label=label)

    plt.title("Accuracy vs. fraction of noise in label data")
    plt.xlabel("Fraction of bits flipped in label training data")
    plt.ylabel("Accuracy on non-noised test set")
    plt.legend()
    plt.show()


def plot_accuracy_over_training_size(tree_depth, num_trees, test_set_size, n_input_features, n_trials):
    """
    Plots how training set size affects model accuracy for one synthetic ground truth function.
    """
    def average_feature_fn(example):
        if sum(example) / len(example) >= 0.5:
            return 1
        else:
            return 0

    # Different training set sizes to test
    train_sizes = [25, 50, 100, 200, 400, 800]
    accuracies = []

    for train_size in train_sizes:
        total_accuracy = 0

        # Average accuracy over multiple trials
        for _ in range(n_trials):
            train_X, train_Y = generate_synthetic_data(train_size, n_input_features, average_feature_fn, 0, 0)
            test_X, test_Y = generate_synthetic_data(test_set_size, n_input_features, average_feature_fn, 0, 0)

            model = RandomForestClassifier(max_depth=tree_depth, n_estimators=num_trees)
            model.fit(train_X, train_Y)

            total_accuracy += evaluate_model(test_X, test_Y, get_accuracy, model)

        accuracies.append(total_accuracy / n_trials)

    # Plot training size against accuracy
    plt.figure()
    plt.plot(train_sizes, accuracies, marker="o")
    plt.title("Accuracy vs. Training Set Size")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy on test set")
    plt.show()


if __name__ == "__main__":

    # Part 1: Groundwork ##

    X, Y = get_titanic_data('titanic_train.csv')
    train_val_X, train_val_Y, test_X, test_Y = split_data(X, Y, 0.8)


    # Part 2: titanic survivorship analysis ##

    tree_depths = [2,4,8,16,32,64]
    num_trees = [1,5,9,15,25,35,45,95]
    num_trials = 5
    plot_model_performance(train_val_X, train_val_Y, test_X, test_Y, tree_depths, num_trees, num_trials)


    # Part 3: Noisy Data Analysis ##

    tree_depth = 12
    num_trees = 5
    train_set_size = 500
    test_set_size = 200
    n_trials = 50
    input_n_features = 12
    plot_accuracy_over_noise(tree_depth, num_trees, train_set_size, test_set_size, input_n_features, n_trials)

    # Part 3.3: DIY Analytics
    plot_accuracy_over_training_size(tree_depth, num_trees, test_set_size, input_n_features, n_trials)
