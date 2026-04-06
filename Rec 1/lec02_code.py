import random
import time
import sys
sys.setrecursionlimit(1_000_000)

## Code from Lecture 1

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

def generate_foods(num_foods, max_val, max_cal):
    names = [f'food{n}' for n in range(num_foods)]
    values = [random.randint(1, max_val) for _ in range(num_foods)]
    calories = [random.randint(1, max_cal) for _ in range(num_foods)]
    return names, values, calories

def test_greedy(foods, calorie_limit, metric):
    metrics = {'value': Item.get_value,
               'cost': lambda x: float('inf') if x.get_cost() == 0
                                 else 1 / x.get_cost(),
               'density': Item.get_density}
    value, solution = knapsack_greedy(foods, calorie_limit, metrics[metric])
    print()
    print(f'Use greedy by {metric} to allocate {calorie_limit} calories')
    print(f'Total value of items taken = {value}')
    for item in solution:
        print(f'   {item}')

def test_greedies(num_items_list, limit):
    for num_items in num_items_list:
        names, values, calories = generate_foods(num_items, 100, 300)
        foods = build_menu(names, values, calories)
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
# test_greedies((10,), 750)
# test_greedies((20,), 750)

## New to Lecture 2

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

# Scenario with 9 items
names = ['wine', 'beer', 'pizza', 'burger', 'fries',
          'cola', 'apple', 'donut', 'cake']
values = [89, 90, 95, 100, 90, 79, 50, 10, 85]
calories = [123, 154, 258, 354, 365, 150, 95, 195, 107]

# # Scenario with 15 items
# names = ['wine', 'beer', 'pizza', 'burger', 'fries',
#           'cola', 'apple', 'donut', 'cake', 'juice',
#           'carrot', 'chocolate', 'celery', 'orings', 'brussels']
# values = [89, 90, 95, 100, 90, 79, 50, 10, 85, 80, 20, 100, 10, 90, 1]
# calories = [123, 154, 258, 354, 365, 150, 95, 195, 107, 39, 25, 406, 15, 190, 38]


# # Specify upper limit on allowable calories
# calorie_limit = 750
# foods = build_menu(names, values, calories)

# for method in (brute_force, decision_tree):
#     solution, value = method(foods, calorie_limit)
#     print(f'{method.__name__} total value of items taken = {value}')
#     for item in solution:
#         print(f'   {item}')


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
        # for item in solution:
        #     print(f'   {item}')

# Try it on examples of increasing size

# start_time = time.time()
# test_funcs(brute_force, (20,), 1000, 100, 300)
# end_time = time.time()
# tot_time = end_time - start_time
# print(f'Elapsed time for brute_force = {tot_time:.6f} seconds')

# start_time = time.time()
# test_funcs(decision_tree, (45,), 1000, 100, 300)
# end_time = time.time()
# tot_time = end_time - start_time
# print(f'Elapsed time for decision_tree = {tot_time:.6f} seconds')

def fib(n):
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

# for i in range(0, 1001):
#     print(f"Try fib({i}) = {fib(i):,}")

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
# 
# print(f'\nfib({i}) is a {len(str(fast_fib(i)))} digit decimal number')

def fast_fib_tab(n):
    # Allocate table for indicies 0 through n
    # Pre-populate with values for fib(0) and fib(1)
    tab = [1] * (n+1)
    for i in range(2, n+1):
        tab[i] = tab[i-1] + tab[i-2]
    return tab[n]

# for i in range(0, 2001, 100):
#     print(f'Try fib({i}) = {fast_fib_tab(i):,}')


# Modified version of decision tree implementation

def knapsack_memo(items, capacity, memo = None):
    """items a list of Item, capacity >= 0
       Solve the 0/1 knapsack problem.
       Return a tuple of an optimal subset of items and their
       total value"""
    global num_calls # for pedagogical reasons
    num_calls += 1
    # Recursively explore tree
    if not items or capacity == 0: # base case
        return (), 0
    if memo == None:
        memo = {}
    if (len(items), capacity) in memo:
        return memo[(len(items), capacity)]
    item = items[0] # Get first item not yet seen
    if item.get_cost() > capacity: # Does current item fit
        return knapsack_memo(items[1:], capacity, memo)
    # Recursively explore consequence of taking current item
    with_item, with_value = knapsack_memo(items[1:], capacity - item.get_cost(),
                                     memo)
    with_item += (item,)
    with_value += item.get_value()
    # Recursively explore consequence of not taking current item
    without_item, without_value = knapsack_memo(items[1:], capacity, memo)
    # Update memo and return better choice
    if with_value > without_value:
        memo[(len(items), capacity)] = (with_item, with_value)
        return with_item, with_value
    else:
        memo[(len(items), capacity)] = (without_item, without_value)
        return without_item, without_value

# # Scenario with 9 items
# names = ['wine', 'beer', 'pizza', 'burger', 'fries',
#           'cola', 'apple', 'donut', 'cake']
# values = [89, 90, 95, 100, 90, 79, 50, 10, 85]
# calories = [123, 154, 258, 354, 365, 150, 95, 195, 107]

# # Scenario with 15 items
# names = ['wine', 'beer', 'pizza', 'burger', 'fries',
#           'cola', 'apple', 'donut', 'cake', 'juice',
#           'carrot', 'chocolate', 'celery', 'orings', 'brussels']
# values = [89, 80, 95, 100, 90, 79, 50, 10, 85, 80, 20, 100, 10, 90, 1]
# calories = [123, 154, 258, 354, 365, 150, 95, 195, 107, 39, 25, 406, 15, 190, 38]


# # Specify upper limit on allowable calories
# calorie_limit = 750
# foods = build_menu(names, values, calories)

# # test_greedies(foods, calorie_limit)

# for method in (brute_force, knapsack_memo):
#     solution, value = brute_force(foods, calorie_limit)
#     print(f'{method.__name__} total value of items taken = {value}')
#     for item in solution:
#         print(f'   {item}')

# Code to test large menus

seed = 1
capacity = 10
max_val = 5
max_cost = 2
num_items = 20
num_calls = 0

# random.seed(seed)
# test_funcs(brute_force, (num_items,), capacity, max_val, max_cost)
# random.seed(seed)
# test_funcs(knapsack_memo, (num_items,), capacity, max_val, max_cost)

# # Try it on examples of increasing size
# random.seed(1)
# for exp in range(1, 11):
#     num_calls = 0
#     num_items = 2**exp
#     test_funcs(knapsack_memo, (num_items,), 1000, 100, 300)
#     print(f'Number of invocations of knap_sack_memo = {num_calls:,}')

# Tabular implementation of knapsack

def knapsack_tabular(items, capacity):
    n = len(items)

    # Initialize the DP table with all 0s,
    # with (n + 1) rows and (capacity + 1) columns
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if items[i-1].get_cost() <= w:  # If current item can be included
                # Max of including the item or not including the item
                dp[i][w] = max(dp[i-1][w],
                               dp[i-1][w-items[i-1].get_cost()] + \
                                   items[i-1].get_value())
            else:
                dp[i][w] = dp[i-1][w]  # Exclude the item

    # Traceback to find the items in the optimal solution
    result_value = dp[n][capacity]
    w = capacity
    selected_items = []

    for i in range(n, 0, -1):
        # If the current item is included in the optimal solution
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(items[i-1])
            w -= items[i-1].get_cost()  # Reduce the remaining weight

    return selected_items, result_value

# # Try it on examples of increasing size
# random.seed(1)
# start_time = time.time()
# for exp in range(1, 11):
#     num_calls = 0
#     num_items = 2**exp
#     test_funcs(knapsack_tabular, (num_items,), 1000, 100, 300)
# end_time = time.time()
# tot_time = end_time - start_time
# print(f'Elapsed time for knapsack_tabular = {tot_time:.6f} seconds')

# for func in (knapsack_memo, knapsack_tabular):
#     random.seed(1)
#     start_time = time.time()
#     test_funcs(func, (1024,), 1000, 100, 300)
#     end_time = time.time()
#     tot_time = end_time - start_time
#     print(f'Elapsed time for {func.__name__} = {tot_time:.6f} seconds')

# def generate_foods(num_foods, max_val, max_cal):
#     names = [f'food{n}' for n in range(num_foods)]
#     values = [random.randint(1, max_val) for _ in range(num_foods)]
#     calories = [random.randint(1, max_cal)*random.random()\
#                 for _ in range(num_foods)]
#     return names, values, calories

# Try it on examples of increasing size
# random.seed(1)
# for exp in range(1, 11):
#     num_calls = 0
#     num_items = 2**exp
#     test_funcs(knapsack_memo, (num_items,), 1000, 100, 300)



