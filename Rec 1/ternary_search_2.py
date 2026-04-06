import numpy as np
import matplotlib.pyplot as plt

def plot_fxn(fxn, min_val, max_val):
    x_values = np.linspace(min_val, max_val, 1000)
    y_values = [fxn(x) for x in x_values]
    plt.plot(x_values, y_values)
    plt.show()

def ternary_search_attempt_a(objective_fcn, min_val, max_val, precision):  
    high, low = min_val, max_val
    # while high and low are too far apart
    while high - low > precision:
        guess1 = low + (high - low) / 3
        guess2 = high - (high - low) / 3
        print(f'min = {guess1:.4f}, max = {guess2:.4f}')
        if objective_fcn(guess1) > objective_fcn(guess2):
            low = guess1 # discard lower third
        else:
            high = guess2 # discard upper third
    return (low + high) / 2

def ternary_search_attempt_b(objective_fcn, min_val, max_val, precision):  
    low, high = min_val, max_val
    # while high and low are too far apart
    while high - low > precision:
        guess2 = low + (high - low) / 3
        guess1 = high - (high - low) / 3
        print(f'min = {guess1:.4f}, max = {guess2:.4f}')
        if objective_fcn(guess1) > objective_fcn(guess2):
            low = guess1 # discard lower third
        else:
            high = guess2 # discard upper third
    return (low + high) / 2

def ternary_search_attempt_c(objective_fcn, min_val, max_val, precision):  
    low, high = min_val, max_val
    # while high and low are too far apart
    while high - low > precision:
        guess1 = low + (high - low) / 3
        guess2 = high - (high - low) / 3
        print(f'min = {guess1:.4f}, max = {guess2:.4f}')
        if objective_fcn(guess1) < objective_fcn(guess2):
            low = guess1 # discard lower third
        else:
            high = guess2 # discard upper third
    return (low + high) / 2

def ternary_search_attempt_d(objective_fcn, min_val, max_val, precision):  
    low, high = min_val, max_val
    # while high and low are too far apart
    while high - low > precision:
        guess1 = low + (high - low) / 3
        guess2 = high - (high - low) / 3
        print(f'min = {guess1:.4f}, max = {guess2:.4f}')
        if objective_fcn(guess1) > objective_fcn(guess2):
            low = guess2 # discard lower third
        else:
            high = guess1 # discard upper third
    return (low + high) / 2

def ternary_search_attempt_secret_fifth_option(objective_fcn, min_val, max_val, precision):  
    low, high = min_val, max_val
    # while high and low are too far apart
    while high - low > precision:
        guess2 = low + (high - low) / 3
        guess1 = high - (high - low) / 3
        print(f'min = {guess1:.4f}, max = {guess2:.4f}')
        if objective_fcn(guess1) > objective_fcn(guess2):
            low = guess2 # discard lower third
        else:
            high = guess1 # discard upper third
    return (low + high) / 2

def objective_fxn(x):
    return 200 - (x - 14)**2

f = objective_fxn

result = ternary_search_attempt_c(f, 0, 30, 0.0001)
print(f"At x = {result:.2f} we get the maximum value {f(result):.2f}")
