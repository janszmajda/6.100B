import random
import time
import math
import matplotlib.pyplot as plt
import sys

# start = time.time()
# end = time.time()
# print(end - start)
# sys.exit()

## Code from Lecture 2

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

def generate_foods(num_foods, max_val, max_cal):
    names = [f'food{n}' for n in range(num_foods)]
    values = [random.randint(1, max_val) for _ in range(num_foods)]
    calories = [random.randint(1, max_cal) for _ in range(num_foods)]
    return names, values, calories

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

# Start of code new to this lecture
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

def test_funcs(func, num_items_list, limit, max_val, max_cost, verbose):
    for num_items in num_items_list:
        print(f'Test {func.__name__} for {num_items} items,'
              f' with capacity {limit}')
        names, values, calories = generate_foods(num_items,
                                                  max_val, max_cost)
        foods = build_menu(names, values, calories)
        random.seed()
        random.shuffle(foods)
        solution, value = func(foods, limit)
        print(f'Total value of items taken = {value}')
        if verbose:
            for item in solution:
                print(f'   {item}')

# capacity, max_val, max_cost, num_items = 10, 5, 2, 20
# seed = 1
# num_calls = 0
# random.seed(seed)
# test_funcs(decision_tree, (num_items,), capacity, max_val, max_cost, True)
# random.seed(seed)
# test_funcs(knapsack_memo, (num_items,), capacity, max_val, max_cost, True)

# sys.exit()

# Try it on examples of increasing size
random.seed(1)
num_calls_list = []
x_vals = []
for exp in range(1, 11):
    num_calls = 0
    num_items = 2**exp
    x_vals.append(num_items)
    test_funcs(knapsack_memo, (num_items,), 1000, 100, 300, verbose = False)
    print(f'Number of invocations of knap_sack_memo = {num_calls:,}')
    num_calls_list.append(num_calls)

plt.plot(x_vals, num_calls_list)
plt.title('Performance of knapsack_memo')
plt.xlabel('Number of Items')
plt.ylabel('Number of Calls')

# sys.exit()

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
                               dp[i-1][w-items[i-1].get_cost()] +\
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

# for func in (knapsack_memo, knapsack_tabular):
#     num_calls = 0
#     random.seed(1)
#     start_time = time.time()
#     test_funcs(func, (1024,), 1000, 100, 300, False)
#     end_time = time.time()
#     tot_time = end_time - start_time
#     print(f'Elapsed time for {func.__name__} = {tot_time:.2f} seconds')

# sys.exit()


def generate_foods(num_foods, max_val, max_cal):
    names = [f'food{n}' for n in range(num_foods)]
    values = [random.randint(1, max_val) for _ in range(num_foods)]
    calories = [random.randint(1, max_cal)*random.random()\
                for _ in range(num_foods)]
    return names, values, calories


# # Try it on examples of increasing size
# random.seed(1)
# for exp in range(1, 11):
#     num_calls = 0
#     num_items = 2**exp
#     test_funcs(knapsack_memo, (num_items,), 1000, 100, 300, verbose = False)
#     print(f'Number of invocations of knap_sack_memo = {num_calls:,}')

# sys.exit()

def get_words():
    word_file = open('words.txt', 'r')
    words = ''
    for w in word_file:
        words += w
    word_list = words.split(' ')
    return word_list

def levenshtein_brute_force(s1, s2):
    # print(f'"{s1}", "{s2}"') # to show what it is doing
    # returns number of edits needed
    if s1 == '':
        return len(s2) # Insert all remaining s2 characters
    if s2 == '':
        return len(s1) # Delete all remaining s1 characters
    if s1[0] == s2[0]: # No increase in distance
        return levenshtein_brute_force(s1[1:], s2[1:])
    else:
        insert = 1 + levenshtein_brute_force(s1, s2[1:])
        delete = 1 + levenshtein_brute_force(s1[1:], s2)
        replace = 1 + levenshtein_brute_force(s1[1:], s2[1:])
        return min(insert, delete, replace)

def test_levenshtein(s1, fcn):
    words = ['out', 'tout', 'route', 'grouch']
    # words = ['an'] # to show a very simple example
    for s2 in words:
        print(f'Distance between {s1} and {s2} is '
              f'{fcn(s1, s2)}')

# test_levenshtein('out', levenshtein_brute_force)

# test_levenshtein('to', levenshtein_brute_force)

sys.exit()

def correct_spelling(alg, max_dist, sentence):
    print(f'Start to correct spelling of "{sentence}"\n')
    dictionary = get_words() # get a list of ~60k allowable words
    print(f'Use a dictionary of {len(dictionary):,} words\n')
    input_words = sentence.split(' ')

    for src in input_words:
        print(f'Check spelling of "{src}"')
        start = time.time()
        min_dist = float('inf')
        is_word = False
        results = []
        for dest in dictionary:
            dist = alg(src, dest)
            if dist == 0:
                is_word = True
                break
            if dist <= min_dist:
                min_dist = dist
                results.append((dest, dist))

        if is_word:
            print(f'"{src}" is in dictionary')
        else:
            if min_dist > max_dist:
                print(f'No words within distance {max_dist} of "{src}"')
            else:
                corrections = [results[i][0] for i in range(len(results))
                   if results[i][1] <= min_dist]
                print(f'Corrections of distance {min_dist} for "{src}" are:')
                print(corrections)
        end = time.time()
        print(f'Time taken = {end - start:.2f} seconds\n')

sentence = 'Ths is a tst of spelimg xpgqrz'
correct_spelling(levenshtein_brute_force, 3, sentence)

sys.exit()

def levenshtein_dp(s1, s2):
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    # Initialize base cases
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    # Fill DP table
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] +
                  (0 if s1[i - 1] == s2[j - 1] else 1))  # Substitution
    return dp[len(s1)][len(s2)]

sentence = 'Ths is a tst of spelimg xpgqrz'
correct_spelling(levenshtein_dp, 3, sentence)
