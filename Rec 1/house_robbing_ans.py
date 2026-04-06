import random

def brute_force(house_vals):

    n = len(house_vals)
    ans = 0

    for b in range(2**n):

        # generates a list of length n, each of which is True or False
        # the Trues represent houses we rob
        houses_to_rob = [bool(b & (1 << i)) for i in range(n)]

        # check that we don't rob two adjacent houses
        valid_plan = True
        for i in range(n-1):
            if houses_to_rob[i] and houses_to_rob[i+1]:
                valid_plan = False

        if valid_plan:
            ans = max(ans, sum([house_vals[i] for i in range(n) if houses_to_rob[i]]))

    return ans

memo = {}

def maximize_money(house_vals):

    if len(house_vals) in memo:
        return memo[len(house_vals)]
    
    if len(house_vals) == 0:
        return 0

    if len(house_vals) == 1:
        return house_vals[0]

    visit_first_house = house_vals[0] + maximize_money(house_vals[2:])
    skip_first_house  = maximize_money(house_vals[1:])

    memo[len(house_vals)] = max(visit_first_house, skip_first_house)
    return memo[len(house_vals)]

random.seed(0)

small_case_1 = [3, 4, 2, 2, 3]
small_case_2 = [1, 4, 1, 2, 3]
small_case_3 = [random.randint(1, 10) for _ in range(15)]
medium_case = [random.randint(1, 10) for _ in range(1000)]

print(maximize_money(medium_case))

# Bonus: optimize the memoization so that the below works!
# large_case = [random.randint(1, 10) for _ in range(100000)]
# print(maximize_money(medium_case))
