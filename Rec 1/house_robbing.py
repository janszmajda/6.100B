import random

random.seed(0)

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
    # fill me in!
    pass

small_case_1 = [1, 4, 1, 2, 3]
small_case_2 = [3, 4, 2, 2, 3]
small_case_3 = [random.randint(1, 10) for _ in range(15)]
large_case = [random.randint(1, 10) for _ in range(1000)]

if __name__ == "__main__":
    assert brute_force(small_case_1) == 7
    assert brute_force(small_case_2) == 8
    assert brute_force(small_case_3) == 56

    print(f"Your result for small case 1 is {maximize_money(small_case_1)}, expecting 7")
    memo = {}
    print(f"Your result for small case 2 is {maximize_money(small_case_2)}, expecting 8")
    memo = {}
    print(f"Your result for small case 3 is {maximize_money(small_case_3)}, expecting 56")
    memo = {}
    print(f"Your result for large case is {maximize_money(large_case)}, expecting 3101")
