################################################################################
# 6.100B Spring 2026
# Problem Set 1
# Name: Jan Szmajda
# Collaborators: PyTutor
# Time: 7
################################################################################

import random
from item import Item

def dp_choose_packing(items, pairs, v_cap, w_cap):
    """
    Chooses an optimal subset of the items to pack in the carry-on and checked
        bags to maximize the total value of the items, while respecting the bags'
        capacity constrains and keeping paired items together.

    Parameters:
        items (list): a list of Item objects
        pairs (list): a list of tuples of Item objects from the `items` list representing
            inseparable pairs of items
        v_cap (int): positive int, the volume capacity of the carry-on bag
        w_cap (int): positive int, the weight capacity of the checked bag

    Returns: a tuple (total_value, carry_on, checked_bag) where
        total_value is the max achievable value of the packed Items
        carry_on and checked_bag are lists of Item objects that achieve the optimal value
    """
    # Pair lookup dictionary
    pair_dict = {}
    for item1, item2 in pairs:
        pair_dict[item1] = item2
        pair_dict[item2] = item1

    memo = {}

    def helper(i, current_v_cap, current_w_cap):
        if i >= len(items) or (current_v_cap == 0 and current_w_cap == 0):  #base case
                return 0, [], []

        if (i, current_v_cap, current_w_cap) in memo:   #memoized?
            return memo[(i, current_v_cap, current_w_cap)]

        item = items[i] #look at first unseen item

        if item not in pair_dict:
            # If no pairs in the problem

            if item.get_volume() > current_v_cap and item.get_weight() > current_w_cap: #checking if item fits anywhere
                return helper(i + 1, current_v_cap, current_w_cap)

            #take item
            #goes in carry
            if not item.cannot_carry() and item.get_volume() <= current_v_cap:
                value1, carry1, check1 = helper(i + 1, current_v_cap - item.get_volume(), current_w_cap)
                carry1 = carry1 + [item]
                value1 += item.get_value()
            else:
                value1 = 0
                carry1 = []
                check1 = []

            #goes in check
            if not item.cannot_check() and item.get_weight() <= current_w_cap:
                value2, carry2, check2 = helper(i + 1, current_v_cap, current_w_cap - item.get_weight())
                check2 = check2 + [item]
                value2 += item.get_value()
            else:
                value2 = 0
                carry2 = []
                check2 = []

            #dont take item
            value3, carry3, check3 = helper(i + 1, current_v_cap, current_w_cap)

            max_value = max([value1, value2, value3])
            if max_value == value1:
                memo[(i, current_v_cap, current_w_cap)] = (value1, carry1, check1)
                return (value1, carry1, check1)
            elif max_value == value2:
                memo[(i, current_v_cap, current_w_cap)] = (value2, carry2, check2)
                return (value2, carry2, check2)
            else:
                memo[(i, current_v_cap, current_w_cap)] = (value3, carry3, check3)
                return (value3, carry3, check3)

        else:
            #if item is a pair
            partner = pair_dict[items[i]]
            partner_index = items.index(partner)

            if partner_index < i:
                return helper(i + 1, current_v_cap, current_w_cap)
            else:
                #1 Both in carry-on (if both fit and both allow carry-on)
                if (not item.cannot_carry() and not partner.cannot_carry()) and (item.get_volume() + partner.get_volume() <= current_v_cap):
                    value4, carry4, check4 = helper(i + 1, current_v_cap - (item.get_volume() + partner.get_volume()), current_w_cap)
                    carry4 = carry4 + [item] + [partner]
                    value4 += item.get_value() + partner.get_value()
                else:
                    value4 = 0
                    carry4 = []
                    check4 = []

                #2 Both in checked (if both fit and both allow checked)
                if (not item.cannot_check() and not partner.cannot_check()) and (item.get_weight() + partner.get_weight() <= current_w_cap):
                    value5, carry5, check5 = helper(i + 1, current_v_cap, current_w_cap - (item.get_weight() + partner.get_weight()))
                    check5 = check5 + [item] + [partner]
                    value5 += item.get_value() + partner.get_value()
                else:
                    value5 = 0
                    carry5 = []
                    check5 = []

                #3 Item i in carry, partner in checked (if capacities/constraints allow)
                if (not item.cannot_carry() and not partner.cannot_check()) and (item.get_volume() <= current_v_cap) and (partner.get_weight() <= current_w_cap):
                    value6, carry6, check6 = helper(i + 1, current_v_cap - item.get_volume(), current_w_cap - partner.get_weight())
                    carry6 = carry6 + [item]
                    check6 = check6 + [partner]
                    value6 += item.get_value() + partner.get_value()
                else:
                    value6 = 0
                    carry6 = []
                    check6 = []

                #4 Item i in checked, partner in carry (if capacities/constraints allow)
                if (not item.cannot_check() and not partner.cannot_carry()) and (item.get_weight() <= current_w_cap) and (partner.get_volume() <= current_v_cap):
                    value7, carry7, check7 = helper(i + 1, current_v_cap - partner.get_volume(), current_w_cap - item.get_weight())
                    carry7 = carry7 + [partner]
                    check7 = check7 + [item]
                    value7 += item.get_value() + partner.get_value()
                else:
                    value7 = 0
                    carry7 = []
                    check7 = []

                #5 Skip both (always valid)
                value8 , carry8, check8 = helper(i + 1, current_v_cap, current_w_cap)

                max_value = max([value4, value5, value6, value7, value8])
                if max_value == value4:
                    memo[(i, current_v_cap, current_w_cap)] = (value4, carry4, check4)
                    return (value4, carry4, check4)
                elif max_value == value5:
                    memo[(i, current_v_cap, current_w_cap)] = (value5, carry5, check5)
                    return (value5, carry5, check5)
                elif max_value == value6:
                    memo[(i, current_v_cap, current_w_cap)] = (value6, carry6, check6)
                    return (value6, carry6, check6)
                elif max_value == value7:
                    memo[(i, current_v_cap, current_w_cap)] = (value7, carry7, check7)
                    return (value7, carry7, check7)
                else:
                    memo[(i, current_v_cap, current_w_cap)] = (value8, carry8, check8)
                    return (value8, carry8, check8)

    # Start the recursion at index 0 with full capacities
    return helper(0, v_cap, w_cap)


if __name__ == "__main__":
    pass

    # Uncomment the section below and delete "pass" above to run the large test case
    # random.seed(21)
    # # 20 random items
    # items = [Item(f'Item {i}', random.randint(5, 20), random.randint(2, 10), random.randint(2, 10)) for i in range(20)]
    # pairs = [(items[6], items[7]), (items[4], items[16])]
    # v_cap = 50
    # w_cap = 50

    # result = dp_choose_packing(items, pairs, v_cap, w_cap)
    # print(result)
