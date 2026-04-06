from lec02_code import test_funcs

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


# Question 8
test_funcs(knapsack_tabular, 
           num_items_list=(100,), 
           limit=300, 
           max_val=100000000, 
           max_cost=300)
