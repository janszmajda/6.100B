import numpy as np
import matplotlib.pyplot as plt
import sys

def find_landing(boat_speed, march_speed, dist_south, dist_east, plot = False):
    times = []
    incr = 1
    miles_south = 0

    while miles_south <= dist_south:
        dist_marched = ((dist_south-miles_south)**2+dist_east**2)**0.5
        tot_time = miles_south/boat_speed + dist_marched/march_speed
        times.append(tot_time)
        miles_south += incr

    if plot:
        plt.plot(times,
              label = f'boat speed = {boat_speed},march_speed = {march_speed}')
        plt.title('Landing vs Time')
        plt.xlabel('Distance Down River')
        plt.ylabel('Total Time (hours)')
        plt.legend()
    best_idx = np.argmin(times)
    landing = best_idx*incr
    print(f'Best landing is {landing} miles south of start')
    dist_marched = ((dist_south-landing)**2+dist_east**2)**0.5
    print(f'Distance marched would be {dist_marched:.2f} miles')
    print(f'Time is {times[best_idx]:.2f} hours')

# plot = False
# find_landing(1, 1, 100, 50, plot)
# find_landing(1.5, 1, 100, 50, plot)
# find_landing(15, 1, 100, 50, plot)

def find_landing_1(boat_speed, march_speed, dist_south, dist_east,
                 des_rate, max_des):
    times = []
    incr = 1
    miles_south = 0
    max_march_dist = max_des*des_rate
    while miles_south <= dist_south:
        dist_marched = ((dist_south-miles_south)**2+dist_east**2)**0.5
        if not dist_marched > max_march_dist:
            tot_time = miles_south/boat_speed + dist_marched/march_speed
            times.append(tot_time)
        else:
            times.append(np.inf)
        miles_south += incr

    best_idx = np.argmin(times)
    landing = best_idx*incr
    print(f'Best landing is {landing} miles south of start')
    dist_marched = ((dist_south-landing)**2+dist_east**2)**0.5
    print(f'Distance marched would be {dist_marched:.2f} miles')
    print(f'Number of desertions is {round(dist_marched*des_rate)}')
    print(f'Time is {times[best_idx]:.2f} hours')

# print('Result with no constraint')
# find_landing(1.5, 1, 100, 50)
# print('\nResult with maximum desertions = 100')
# find_landing_1(1.5, 1, 100, 50, 1, 100)
# print('\nResult with maximum desertions = 55')
# find_landing_1(1.5, 1, 100, 50, 1, 55)

def ternary_search(objective_fcn, min_val, max_val, precision):
    low, high = min_val, max_val
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

def time_taken(boat_speed, march_speed, dist_south, dist_east,
               landing):
    dist_marched = ((dist_south-landing)**2+dist_east**2)**0.5
    return landing/boat_speed + dist_marched/march_speed

# objective_fcn = lambda landing: time_taken(1.5, 1, 100, 50,
#                                            landing)
# landing = ternary_search(objective_fcn, 0, 100, 0.01)
# tot_time = objective_fcn(landing)
# print(f'Best landing is {landing:.4f} miles south of start')
# print(f'Time is {tot_time:.2f} hours')

########################################
# Representation of individual food items and problem inputs
########################################


class Item(object):
    def __init__(self, n, v, w):
        self._name = n
        self._value = v
        self._calories = w
    def get_value(self):
        return self._value
    def get_cost(self):
        return self._calories
    def get_density(self):
        if self._calories > 0:
            return self._value / self._calories
        else:
            return float('inf')
    def __str__(self):
        return f'{self._name}: <{self._value}, {self._calories}>'


def build_menu(names, values, calories):
    """names, values, calories are lists of same length
       names are strings, values and calories are non-negative numbers
       returns a list of Items"""
    menu = []
    for i in range(len(values)):
        menu.append(Item(names[i], values[i], calories[i]))
    return menu


# Scenario with 9 items
names = ['wine', 'beer', 'pizza', 'burger', 'fries',
          'cola', 'apple', 'donut', 'cake']
values = [89, 90, 95, 100, 90, 79, 50, 10, 85]
calories = [123, 154, 258, 354, 365, 150, 95, 195, 107]

def generate_combinations(n):
    """Assumes n is a non-negative integer.
    Returns a list containing all binary strings of length n."""
    if n == 0:
        return []
    else:
        return [format(i, f'0{n}b') for i in range(2**n)]

def test_combinations():
    test_cases = [0, 1, 2, 3]
    expected_results = [[], ['0', '1'], ['00', '01', '10', '11'],
        ['000', '001', '010', '011', '100', '101', '110', '111']]

    for n, expected in zip(test_cases, expected_results):
        actual = generate_combinations(n)
        if actual == expected:
            print(f"Test for n = {n} passed: {actual}")
        else:
            print(f'Test for n = {n} failed.',
                  f'Expected: {expected}, Actual: {actual}')

# test_combinations()


########################################
# Brute force solution
########################################

def brute_force(items, capacity):
    """items a list, capacity >= 0
       solves the 0/1 knapsack problem with items and capacity
       returns as a list a subset of items that don't exceed cost_limit
       while maximizing total value, and also the value of that list"""
    n = len(items)
    all_combinations = generate_combinations(n)
    best_value = 0
    best_subset = []
    for combination in all_combinations:
        subset = [items[i] for i in range(n) if combination[i] == '1']
        subset_cost = sum([item.get_cost() for item in subset])
        if subset_cost <= capacity:
            subset_value = sum([item.get_value() for item in subset])
            if subset_value > best_value:
                best_value = subset_value
                best_subset = subset
    return best_subset, best_value

# Scenario with 9 items
names = ['wine', 'beer', 'pizza', 'burger', 'fries',
          'cola', 'apple', 'donut', 'cake']
values = [89, 90, 95, 100, 90, 79, 50, 10, 85]
calories = [123, 154, 258, 354, 365, 150, 95, 195, 107]

# Scenario with 15 items
names = ['wine', 'beer', 'pizza', 'burger', 'fries',
          'cola', 'apple', 'donut', 'cake', 'juice',
          'carrot', 'chocolate', 'celery', 'orings', 'brussels']
values = [89, 90, 95, 100, 90, 79, 50, 10, 85, 80, 20, 100, 10, 90, 1]
calories = [123, 154, 258, 354, 365, 150, 95, 195, 107, 39, 25, 406, 15, 190, 38]

calorie_limit = 750
foods = build_menu(names, values, calories)
solution, value = brute_force(foods, calorie_limit)
print(f'Total value of items taken = {value}')
for item in solution:
    print(f'   {item}')
