import matplotlib.pyplot as plt
import numpy as np
import random
import sys

#
# Set up some global plot parameters to make plots look better
# set line width
plt.rcParams['lines.linewidth'] = 3
# set font size for titles
plt.rcParams['axes.titlesize'] = 15
# set font size for labels on axes
plt.rcParams['axes.labelsize'] = 15
# set size of numbers on x-axis
plt.rcParams['xtick.labelsize'] = 15
# set size of numbers on y-axis
plt.rcParams['ytick.labelsize'] = 15
# set size of ticks on x-axis
plt.rcParams['xtick.major.size'] = 7
# set size of ticks on y-axis
plt.rcParams['ytick.major.size'] = 7
# set size of markers
plt.rcParams['lines.markersize'] = 8
# set values for legend
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.handlelength'] = 0.5


def get_data(filename):
    with open(filename, 'r') as data_file:
        y_vals = []
        x_vals = []
        data_file.readline() # discard header
        for line in data_file:
            y, x = line.split()
            y_vals.append(float(y))
            x_vals.append(float(x))
        return np.array(x_vals), np.array(y_vals)

##########################################
# Plot spring data and fit a line to it


# x_vals, y_vals = get_data('springData.txt')
# x_vals = x_vals*9.81 # force due to gravity
# plt.figure()
# plt.plot(x_vals, y_vals, 'bo', label='Measured displacements')
# plt.xlabel('|Force| (Newtons)')
# plt.ylabel('Distance (meters)')
# plt.ylim(0, 0.5)
# plt.title('Measured Displacement of Spring')
# plt.legend(loc = 'lower right')

# # sys.exit()

# # # Try some random lines
# random.seed(3)
# i1, i2 = random.sample(range(len(x_vals)), 2)
# x1, y1 = x_vals[i1], y_vals[i1]
# x2, y2 = x_vals[i2], y_vals[i2]
# k = (y2-y1) / (x2-x1)
# b = y1 - k*x1
# y_pred = k*x_vals + b
# plt.plot(x_vals, y_pred, 'r', label=f'Linear fit, k = {k:.3f}')
# plt.legend()

# a, b = np.polyfit(x_vals, y_vals, 1)
# y_pred = a*x_vals + b
# k = 1/a
# print(f'a = {a:.3f}, b = {b:.3f}')
# plt.plot(x_vals, y_pred, 'r', label=f'Linear fit, k = {k:.3f}')
# plt.legend()

# # Will return to this later in lecture
# model = np.polyfit(x_vals, y_vals, 1)
# y_pred = np.polyval(model, x_vals)
# k = 1/model[0]
# print(f'a = {model[0]:.3f}, b = {model[1]:.3f}')
# plt.plot(x_vals, y_pred, 'r', label=f'Linear fit, k = {k:.3f}')
# plt.legend()

# sys.exit()

##########################################
# Evaluate different degree models for mystery data


# x_vals, y_vals = get_data('mysteryData.txt')
# plt.figure()
# plt.plot(x_vals, y_vals, 'o', label='Data Points')
# plt.xlabel('Independent Variable')
# plt.ylabel('Dependent Variable')
# plt.ylim(-100, 350)
# plt.title('Mystery Data')

# model1 = np.polyfit(x_vals, y_vals, 1)
# plt.plot(x_vals, np.polyval(model1, x_vals), label='Linear Model')
# model2 = np.polyfit(x_vals, y_vals, 2)
# plt.plot(x_vals, np.polyval(model2, x_vals), 'r--', label='Quadratic Model')
# plt.legend()

# sys.exit()


def mean_squared_error(data, predicted):
    error = 0.0
    for i in range(len(data)):
        error += (data[i] - predicted[i])**2
    return error/len(data)


# y_pred = np.polyval(model1, x_vals)
# print(f'Mean squared error for linear model = '
#       f'{mean_squared_error(y_vals, y_pred):.2f}')

# y_pred = np.polyval(model2, x_vals)
# print('Mean squared error for quadratic model = '
#       f'{mean_squared_error(y_vals, y_pred):.2f}')

# sys.exit()


def r_squared(observed, predicted):
    error = ((predicted - observed)**2).sum()
    mean_error = error / len(observed)
    return 1 - mean_error / np.var(observed)


def gen_fits(x_vals, y_vals, degrees):
    models = []
    for d in degrees:
        model = np.polyfit(x_vals, y_vals, d)
        models.append(model)
        print(model)
    return models


def test_fits(models, degrees, x_vals, y_vals, title):
    plt.figure()
    plt.plot(x_vals, y_vals, 'o', label='Data')
    for i in range(len(models)):
        y_pred = np.polyval(models[i], x_vals)
        error = r_squared(y_vals, y_pred)
        plt.plot(x_vals, y_pred, label=f'Fit of degree {degrees[i]}, R2 = {error:.3f}')
    plt.legend()
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.title(title)

# x_vals, y_vals = get_data('mysteryData.txt')
# degrees = (1, 2)
# # degrees = (2, 4, 8, 16)
# models = gen_fits(x_vals, y_vals, degrees)
# test_fits(models, degrees, x_vals, y_vals, 'Mystery Data')

# sys.exit()

# x = [-1, 0, 1, 2]
# y = [2, -5, -8, 9]
# model = np.polyfit(x, y, 1)
# print(r_squared(y, np.polyval(model, x)))
# model = np.polyfit(x, y, 2)
# print(r_squared(y, np.polyval(model, x)))
# model = np.polyfit(x, y, 3)
# print(r_squared(y, np.polyval(model, x)))
# model = np.polyfit(x, y, 4)
# print(r_squared(y, np.polyval(model, x)))
# model = np.polyfit(x, y, 5)
# print(r_squared(y, np.polyval(model, x)))


##########################################
# Run trained model on testing data


def gen_noisy_parabolic_data(a, b, c, x_vals, filename):
    y_vals = []
    for x in x_vals:
        theoretical_val = a*x**2 + b*x + c
        y_vals.append(theoretical_val + random.gauss(0, 35))
    with open(filename,'w') as f:
        f.write('y        x\n')
        for i in range(len(y_vals)):
            f.write(str(y_vals[i]) + ' ' + str(x_vals[i]) + '\n')

def gen_noisy_parabolic_data(a, b, c, x_vals, filename, plot = False):
    theoretical_vals, noise_vals = [], []
    for x in x_vals:
        theoretical_vals.append(a*x**2 + b*x + c)
        noise_vals.append(random.gauss(0, 35))
    y_vals = np.array(theoretical_vals) + np.array(noise_vals)
    with open(filename,'w') as f:
        f.write('y        x\n')
        for i in range(len(y_vals)):
            f.write(str(y_vals[i]) + ' ' + str(x_vals[i]) + '\n')
    if plot:
        plt.plot(theoretical_vals, label = 'signal')
        plt.plot(noise_vals, label = 'noise')
        plt.plot(y_vals, label = 'signal + noise')
        plt.legend()
        print(sum(noise_vals)/len(noise_vals))


# random.seed(0)
# x_vals = range(-10, 11, 1)
# a, b, c = 3, 0, 0
# gen_noisy_parabolic_data(a, b, c, x_vals, 'parabola1.txt', True)
# plt.figure()
# gen_noisy_parabolic_data(a, b, c, x_vals, 'parabola2.txt', True)

# degrees = (1, 2, 16)

# x_vals1, y_vals1 = get_data('parabola1.txt')
# models1 = gen_fits(x_vals1, y_vals1, degrees)
# test_fits(models1, degrees, x_vals1, y_vals1, 'Parabola 1')

# x_vals2, y_vals2 = get_data('parabola2.txt')
# models2 = gen_fits(x_vals2, y_vals2, degrees)
# test_fits(models2, degrees, x_vals2, y_vals2, 'Parabola 2')

# test_fits(models1, degrees, x_vals2, y_vals2, 'Apply Parabola 1 Model to Parabola 2')

# sys.exit()


def split_data(x_vals, y_vals, frac_training, plot=True):
    training_size = int(len(x_vals)*frac_training)
    training_indices = random.sample(range(len(x_vals)), training_size)
    training_x, training_y, test_x, test_y = [], [], [], []
    for i in range(len(x_vals)):
        if i in training_indices:
            training_x.append(x_vals[i])
            training_y.append(y_vals[i])
        else:
            test_x.append(x_vals[i])
            test_y.append(y_vals[i])
    if plot:
        plt.plot(training_x, training_y, '.', label='Training')
        plt.plot(test_x, test_y, '.', label='Test')
        plt.legend()
        plt.title('Training and Test Splits')
    return (training_x, training_y), (test_x, test_y)


def fit_and_validate(x_vals, y_vals, degrees):
    training, test = split_data(x_vals, y_vals, 0.5)
    models = []
    for d in degrees:
        models.append(np.polyfit(training[0], training[1], d))
    for m in models:
        print([round(c, 2) for c in m])
    test_fits(models, degrees, training[0], training[1], 'Fit to Training Data')
    test_fits(models, degrees, test[0], test[1], 'Applied to Test Data')

# random.seed(0)
# x_vals = range(-20, 20, 1)
# a, b, c = 3, 0, 0
# gen_noisy_parabolic_data(a, b, c, x_vals, 'parabola.txt')

# x_vals, y_vals = get_data('parabola.txt')
# degrees = (2, 16)
# fit_and_validate(x_vals, y_vals, degrees)


# random.seed(0)
# x_vals, y_vals = get_data('parabola.txt')
# degrees = (1, 2, 4, 8, 16)
# num_trials = 200
# for d in degrees:
#     r2_vals = []
#     for _ in range(num_trials):
#         training, test = split_data(x_vals, y_vals, 0.5, plot=False)
#         model = np.polyfit(training[0], training[1], d)
#         test_pred = np.polyval(model, test[0])
#         r2 = r_squared(test[1], test_pred)
#         r2_vals.append(r2)
#     r2_mean = np.mean(r2_vals)
#     print(f'Fit of degree {d}, mean R2 = {r2_mean}')

# sys.exit()


############################################################
# PROBABILITY DISTRIBUTIONS
############################################################


########################################
# Discrete Uniform distribution

# num_samples = 1000
# random.seed(0)
# a, b = 2, 11
# for trial in range(2):
#     uniform_vals = [random.randint(a, b) for _ in range(num_samples)]
#     counts, bins = np.histogram(uniform_vals, bins = b-a+1)
#     prob_dist = counts/num_samples
#     x_vals = [(bins[i] + bins[i-1])/2 for i in range(1, len(bins))]
#     plt.bar(x_vals, prob_dist, label = f'trial {trial}', alpha = 0.5)
# plt.xlabel('Outcome value')
# plt.xlim(1, 12)
# plt.xticks(range(1, 13))
# plt.ylabel('Probability density')
# plt.title(f'Discrete uniform distribution on [{a}, {b}]')
# plt.legend()


########################################
# Normal/Gaussian distribution

# mu, sigma = 20, 3
# num_samples = 100_000
# num_bins = 25
# normal_vals = [random.gauss(20, 3) for _ in range(num_samples)]
# counts, bins = np.histogram(normal_vals, bins = num_bins)
# prob_dist = counts/num_samples
# x_vals = [(bins[i] + bins[i-1])/2 for i in range(1, len(bins))]
# plt.bar(x_vals, prob_dist)
# plt.xlabel('Outcome value')
# plt.ylabel('Probability density')
# plt.title(f'Normal distribution with $\mu={mu}$ and $\sigma = {sigma}$')


########################################
# Exponential distribution

# lam = 0.5
# num_samples = 100_000
# num_bins = 25
# exp_vals = [random.expovariate(lam) for _ in range(num_samples)]
# counts, bins = np.histogram(exp_vals, bins = num_bins)
# prob_dist = counts/num_samples
# x_vals = [(bins[i] + bins[i-1])/2 for i in range(1, len(bins))]
# plt.bar(x_vals, prob_dist)
# plt.xlim(0, 15)
# plt.xlabel('Outcome value')
# plt.ylabel('Probability density')
# plt.title(f'Exponential distribution with $\lambda={lam}$')


########################################
# Bernoulli distribution

# num_samples = 1_000
# num_bins = 2
# prob = 0.3
# bern_vals = [random.choices([0, 1], weights=[1-prob, prob], k=num_samples)]
# counts, bins = np.histogram(bern_vals, bins = num_bins)
# prob_dist = counts/num_samples
# plt.bar((0, 1), prob_dist)
# plt.xticks((0, 1))
# plt.xlabel('Outcome value')
# plt.ylabel('Probability density')
# plt.title(f'Bernoulli distribution with prob = {prob}')

# sys.exit()

# ########################################
# # Binomial distribution

# num_samples = 1_000
# num_trials = 1_000
# num_bins = 11
# prob = 0.3
# binom = np.array(random.choices([0, 1], weights=[1-prob, prob], k=num_samples))
# for i in range(num_trials - 1):
#     bern = np.array(random.choices([0, 1], weights=[1-prob, prob], k=num_samples))
#     binom += bern

# print(f'Mean = {sum(binom)/len(binom)}, Std = {np.std(binom)}')

# counts, bins = np.histogram(binom, bins = num_bins)
# bin_centers = np.round((bins[:-1] + bins[1:])/2)
# prob_dist = counts/num_samples
# plt.bar(bin_centers, prob_dist)
# plt.xticks(bin_centers, rotation = 45)
# plt.xlabel('Outcome value')
# plt.ylabel('Probability density')
# plt.title(f'Binomial distribution with prob = {prob}\n'
#           f'|Sample size| = {num_samples}, |Trials| = {num_trials}')

# sys.exit()



# A demo showing that many random binary choices leads to a normal distribution
# %matplotlib qt  # Commented out - only works in Jupyter/IPython
from matplotlib.animation import FuncAnimation

def galton_board(num_slots, num_balls):
    slots = [0] * (num_slots)  # Create slots to collect balls

    for _ in range(num_balls):
        pos = 0  # Start at the top of the board
        for _ in range(num_slots): # number of levels
            if random.random() < 0.5:  # 50% chance to move right
                pos += 1
        slots[pos] += 1  # Increment the corresponding slot

        yield slots

def update(frame):
    plt.cla()
    plt.bar(range(len(frame)), frame)
    plt.xlabel('Slots')
    plt.ylabel('Number of Balls')
    plt.title(f'Galton Board Simulation with {num_balls:,} balls')

random.seed(1)

num_slots = 10  # One more than number of slots in the Galton board
num_balls = 10  # Number of balls to drop
num_balls = 1000  # Number of balls to drop
if num_balls < 50:
    interval = 500
else:
    interval = 0.00001
results = galton_board(num_slots, num_balls)

fig = plt.figure()
ani = FuncAnimation(fig, update, frames=results, interval=interval,
                    cache_frame_data=False, repeat=False)
plt.show()
