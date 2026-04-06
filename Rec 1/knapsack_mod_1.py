from lec02_code import test_funcs
from lec02_code import knapsack_tabular

memo = {}
def knapsack_memo(items, capacity):
    """items: a list of Item, capacity >= 0

       Solve the 0/1 knapsack problem.
       Return a tuple of an optimal subset of items and their
       total value"""

    # Recursively explore tree
    if not items or capacity == 0: # base case
        return (), 0

    if (len(items), capacity) in memo:
        return memo[(len(items), capacity)]

    item = items[0] # Get first item not yet seen
    if item.get_cost() > capacity: # Does current item fit
        return knapsack_memo(items[1:], capacity)

    # Recursively explore consequence of taking current item
    with_item, with_value = knapsack_memo(items[1:], capacity - item.get_cost())
    with_item += (item,)
    with_value += item.get_value()

    # Recursively explore consequence of not taking current item
    without_item, without_value = knapsack_memo(items[1:], capacity)

    # Update memo and return better choice
    if with_value > without_value:
        memo[(len(items), capacity)] = (with_item, with_value)
        return with_item, with_value
    else:
        memo[(len(items), capacity)] = (without_item, without_value)
        return without_item, without_value

# Question 3
# test_funcs(knapsack_memo, 
#            num_items_list=(5,), 
#            limit=10000000,    # my calorie limit
#            max_val=30,  # max happiness i get when i eat the food
#            max_cost=100000000) # max calories for each food item

# Question 4
# test_funcs(knapsack_tabular, 
#            num_items_list=(5,), 
#            limit=100000000, 
#            max_val=30, 
#            max_cost=1000000)

# Question 5
# test_funcs(knapsack_memo, 
#            num_items_list=(100,), 
#            limit=100000000, 
#            max_val=30, 
#            max_cost=1000000)

# Question 7
# test_funcs(knapsack_memo, 
#            num_items_list=(100,), 
#            limit=300, 
#            max_val=100000000, 
#            max_cost=300)

# Question 8
# test_funcs(knapsack_tabular, 
#            num_items_list=(100,), 
#            limit=30, 
#            max_val=100000000, 
#            max_cost=30)
