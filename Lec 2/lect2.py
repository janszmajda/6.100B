import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys


########################################
# Representation of individual food items and problem inputs
# Same as last lecture
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

def generate_combinations(n):
    """Assumes n is a non-negative integer.
    Returns a list containing all binary strings of length n."""
    if n == 0:
        return []
    else:
        return [format(i, f'0{n}b') for i in range(2**n)]

########################################
# Brute force solution from last lecture
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

# # Scenario with 9 items
# names = ['wine', 'beer', 'pizza', 'burger', 'fries',
#           'cola', 'apple', 'donut', 'cake']
# values = [89, 90, 95, 100, 90, 79, 50, 10, 85]
# calories = [123, 154, 258, 354, 365, 150, 95, 195, 107]

# calorie_limit = 750
# foods = build_menu(names, values, calories)
# solution, value = brute_force(foods, calorie_limit)
# print(f'Total value of items taken = {value}')
# for item in solution:
#     print(f'   {item}')

# sys.exit()

########################################
# Start of new material
########################################

def generate_foods(num_foods, max_val, max_cal):
    names = [f'food{n}' for n in range(num_foods)]
    values = [random.randint(1, max_val) for _ in range(num_foods)]
    calories = [random.randint(1, max_cal) for _ in range(num_foods)]
    return names, values, calories

def test_funcs(func, num_items_list, limit, max_val, max_cost):
    for num_items in num_items_list:
        print(f'Test {func.__name__} for {num_items} items,'
              f' {limit} max')
        random.seed(0)
        names, values, calories = generate_foods(num_items,
                                                 max_val, max_cost)
        foods = build_menu(names, values, calories)
        random.seed()
        random.shuffle(foods)
        solution, value = func(foods, limit)
        print(f'Total value of items taken = {value}')
        for item in solution:
            print(f'   {item}')

# test_funcs(brute_force, (10, 20, 30), 750, 100, 300)

# sys.exit()

def decision_tree(items, capacity):
    """items a list of Item, capacity >= 0
       Solve the 0/1 knapsack problem.
       Return a tuple of an optimal subset of items and their
       total value"""
    # Recursively explore tree
    if not items or capacity == 0: # base case
        return (), 0
    item = items[0] # Get first item not yet seen
    if item.get_cost() > capacity: # Does current item fit
        return decision_tree(items[1:], capacity)
    # Recursively explore consequence of taking current item
    with_item, with_value = decision_tree(items[1:], capacity - item.get_cost())
    with_item += (item,)
    with_value += item.get_value()
    # Recursively explore consequence of not taking current item
    without_item, without_value = decision_tree(items[1:], capacity)
    # Update using better choice
    if with_value > without_value:
        return with_item, with_value
    else:
        return without_item, without_value

# # Scenario with 9 items

# names = ['wine', 'beer', 'pizza', 'burger', 'fries',
#           'cola', 'apple', 'donut', 'cake']
# values = [89, 90, 95, 100, 90, 79, 50, 10, 85]
# calories = [123, 154, 258, 354, 365, 150, 95, 195, 107]

# calorie_limit = 750
# foods = build_menu(names, values, calories)
# solution, value = brute_force(foods, calorie_limit)
# print(f'Total value of items taken with brute force = {value}')
# for item in solution:
#     print(f'   {item}')

# solution, value = decision_tree(foods, calorie_limit)
# print(f'Total value of items taken with decision tree = {value}')
# for item in solution:
#     print(f'   {item}')

# sys.exit()

# Try it on examples of increasing size

# start_time = time.time()
# test_funcs(brute_force, (20,), 1000, 100, 300)
# end_time = time.time()
# tot_time = end_time - start_time
# print(f'Elapsed time for brute_force = {tot_time:.2f} seconds')

# start_time = time.time()
# test_funcs(decision_tree, (20,), 1000, 100, 300)
# end_time = time.time()
# tot_time = end_time - start_time
# print(f'Elapsed time for decision_tree = {tot_time:.2f} seconds')

# sys.exit()

# start_time = time.time()
# test_funcs(decision_tree, (30,), 1000, 100, 300)
# end_time = time.time()
# tot_time = end_time - start_time
# print(f'Elapsed time for decision_tree = {tot_time:.2f} seconds')

# start_time = time.time()
# test_funcs(decision_tree, (40,), 1000, 100, 300)
# end_time = time.time()
# tot_time = end_time - start_time
# print(f'Elapsed time for decision_tree = {tot_time:.2f} seconds')

# start_time = time.time()
# test_funcs(decision_tree, (45,), 1000, 100, 300)
# end_time = time.time()
# tot_time = end_time - start_time
# print(f'Elapsed time for decision_tree = {tot_time:.2f} seconds')

# sys.exit()

def knapsack_greedy(items, capacity):
    """items a list of Item, capacity a non-negative number
    Uses a greedy algorithm to provide an approximation of
    a solution to the 0/1 knapsack problem.
    Returns a list of items and the value of that list."""
    items_sorted = sorted(items,
                          key=lambda item: item.get_density(),
                          reverse=True)
    knapsack = []
    total_value = 0
    for item in items_sorted:
        if item.get_cost() <= capacity:
            knapsack.append(item)
            capacity -= item.get_cost()
            total_value += item.get_value()
    return knapsack, total_value

# start_time = time.time()
# test_funcs(decision_tree, (40,), 1000, 100, 300)
# end_time = time.time()
# tot_time = end_time - start_time
# print(f'Elapsed time for decision_tree = {tot_time:.2f} seconds')

# start_time = time.time()
# test_funcs(knapsack_greedy, (40,), 1000, 100, 300)
# end_time = time.time()
# tot_time = end_time - start_time
# print(f'Elapsed time for knapsack_greedy = {tot_time:.2f} seconds')

# sys.exit()

def knapsack_greedy(items, capacity,
                    key=lambda x: x.get_density()):
    """items a list of Item, capacity a non-negative number
    Uses a greedy algorithm to provide an approximation of
    a solution to the 0/1 knapsack problem.
    Returns a list of items and the value of that list."""
    items_copy = sorted(items, key=key, reverse=True)
    result = []
    total_value, total_weight = 0, 0
    for item in items_copy:
        if total_weight + item.get_cost() <= capacity:
            result.append(item)
            total_weight += item.get_cost()
            total_value += item.get_value()
    return result, total_value

def test_greedies(num_items_list, limit, compare = False):
    for num_items in num_items_list:
        names, values, calories = generate_foods(num_items, 100, 300)
        foods = build_menu(names, values, calories)
        if compare:
            optimum = decision_tree(foods, limit)[1]
            print(f'Optimal value = {optimum}')
        metrics = {'value': Item.get_value,
                   'cost': lambda x: float('inf') if x.get_cost() == 0
                                     else 1 / x.get_cost(),
                   'density': Item.get_density}
        for key in metrics:
            print(f'Use key {key}')
            func = lambda foods, limit: knapsack_greedy(foods, limit,
                                                        metrics[key])
            solution, value = func(foods, limit)
            print(f'Total value of items taken = {value}')
            for item in solution:
                print(f'   {item}')

# random.seed(1)
# test_greedies((10,), 750, compare = True)

# random.seed(2)
# test_greedies((10,), 750, True)

# random.seed(3)
# test_greedies((10,), 750, True)

# sys.exit()

num_items = 1_000_000
start_time = time.time()
test_funcs(knapsack_greedy, (num_items,), 10_000, 100, 300)
end_time = time.time()
tot_time = end_time - start_time
print(f'Elapsed time for knapsack_greedy with {num_items:,} = '
      f'{tot_time:.2f} seconds')

# sys.exit()

def fib(n):
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

for i in range(0, 2001):
    print(f"Try fib({i}) = {fib(i):,}")

# sys.exit()

def fast_fib(n, memo = None):
    if memo == None:
        memo = {}
    if n == 0 or n == 1:
        return 1
    try:
        return memo[n]
    except KeyError:
        result = fast_fib(n-1, memo) + fast_fib(n-2, memo)
        memo[n] = result
        return result

# for i in range(0, 2001, 100):
#     print(f'Try fib({i}) = {fast_fib(i):,}')

# print(f'\nfib({i}) is a {len(str(fast_fib(i)))} digit decimal number')

def fast_fib_tab(n):
    # Allocate table for indicies 0 through n
    # Pre-populate with values for fib(0) and fib(1)
    tab = [1] * (n+1)
    for i in range(2, n+1):
        tab[i] = tab[i-1] + tab[i-2]
    return tab[n]

for i in range(0, 2001, 100):
    print(f'Try fib({i}) = {fast_fib_tab(i):,}')
