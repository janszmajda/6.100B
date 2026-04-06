import copy
import unittest
from ps1 import *
from item import Item

def check_valid_packing(total_value, carry_on, checked_bag, items, pairs, v_cap, w_cap):
    """
    total_value, carry_on, checked_bag: output of choose_packing
    items, pairs, v_cap, w_cap: input to choose_packing
    Asserts that the packing respects the weight and volume constraints,
        pairs are handled correctly,
        the total value is correct,
        and each item appears at most once in the packing.
    """
    v_used = sum(item.get_volume() for item in carry_on)
    w_used = sum(item.get_weight() for item in checked_bag)

    # check each item appears at most once
    assert len(set(carry_on).intersection(set(checked_bag))) == 0 and len(set(carry_on + checked_bag)) == len(carry_on) + len(checked_bag), \
        "Packing not valid: an item appears more than once in the packing."

    # check that paired items are handled correctly
    packed_items = set(carry_on + checked_bag)
    for (item1, item2) in pairs:
        assert (item1 in packed_items) == (item2 in packed_items), \
        f"Packing not valid: must include either both items or neither item in the pair ({item1.get_name()}, {item2.get_name()})"

    # check weight and volume constraints
    assert v_used <= v_cap, "Packing not valid: carry-on volume exceeded: %s > %s" % (v_used, v_cap)
    assert w_used <= w_cap, "Packing not valid: checked bag weight exceeded: %s > %s" % (w_used, w_cap)

    # check items in carry-on are actually carry-able
    assert all(not item.cannot_carry() for item in carry_on), \
            "Packing not valid: item %s in carry-on but cannot be carried" \
            % [item for item in carry_on if not item.cannot_carry()][0]

    # check items in check-in are actually check-able
    assert all(not item.cannot_check() for item in checked_bag), \
            "Packing not valid: item %s in check-in but cannot be checked in" \
            % [item for item in checked_bag if not item.cannot_check()][0]

    # check total value is correct
    value = sum(item.get_value() for item in carry_on) + sum(item.get_value() for item in checked_bag)
    assert value == total_value, "Total value does not match items packed: %s != %s" % (value, total_value)

def check_packing_returntype(result):
    assert isinstance(result, tuple), "choose_packing didn't return a tuple: instead returned an instance of %s." % type(result)
    assert len(result) == 3, "choose_packing didn't return 3 elements (total_value, carryon, checked). Expected %s, got %s." % (3, len(result))
    assert isinstance(result[0], int), "choose_packing's first return value (total_value) should be an int: instead got %s." % type(result[0])

    # check carryon is a list...
    assert isinstance(result[1], list), "choose_packing's second return value (carryon) should be a list of Items: instead got %s." % type(result[1])
    # ...of Items
    for item in result[1]:
        assert isinstance(item, Item), f"choose_packing's second return value (carryon) should be a list of Items: element {item} is instead a {type(item)}"

    # check checked is a list...
    assert isinstance(result[2], list), "choose_packing's third return value (checked) should be a list of Items: instead got %s." % type(result[1])
    # ...of Items
    for item in result[2]:
        assert isinstance(item, Item), f"choose_packing's third return value (checked) should be a list of Items: element {item} is instead a {type(item)}"


class TestPS1(unittest.TestCase):
    def test_dp_choose_packing_all_fit_no_pairs(self):
        # test case where all items fit, no constraints, no pairs
        items = [
            Item('shampoo', 10, 5, 10),
            Item('Introduction to Computation and Programming Using Python, Third Edition, With Application to Computational Modeling and Understanding Data by John Guttag', 15, 10, 20),
            Item('flavored rice cakes', 7, 3, 5),
            Item('jorts', 8, 2, 8),
        ]

        pairs = []

        v_cap = 11
        w_cap = 24

        result = dp_choose_packing(items, pairs, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, pairs, v_cap, w_cap)
        self.assertEqual(result[0], 40) # check for optimality

    def test_dp_choose_packing_not_all_fit_no_pairs(self):
        # test case where not all items fit, no constraints, no pairs
        items = [
            Item('shampoo', 10, 5, 10),
            Item('Introduction to Computation and Programming Using Python, Third Edition, With Application to Computational Modeling and Understanding Data by John Guttag', 15, 7, 20),
            Item('flavored rice cakes', 7, 3, 5),
            Item('jorts', 8, 5, 15),
        ]

        pairs = []

        v_cap = 11
        w_cap = 24

        result = dp_choose_packing(items, pairs, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, pairs, v_cap, w_cap)
        self.assertEqual(result[0], 33) # check for optimality

    def test_dp_choose_packing_all_fit_with_pairs(self):
        # test case where all items fit, no constraints, with pairs
        items = [
            Item('rubber duck', 10, 5, 10),
            Item('favorite jeans', 15, 10, 20),
            Item('notebook', 7, 3, 5),
            Item('candy from lecture', 8, 2, 8),
            Item('toothbrush', 5, 5, 2),
            Item('toothpaste', 5, 2, 5),
        ]

        pairs = [(items[4], items[5])]

        v_cap = 15
        w_cap = 25
        result = dp_choose_packing(items, pairs, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, pairs, v_cap, w_cap)
        self.assertEqual(result[0], 50) # check for optimality

    def test_dp_choose_packing_first_not_optimal(self):
        # test case where it's not optimal to take the first item
        items = [
            Item('bucket hat collection', 8, 6, 6, cannot_carry=True),
            Item('spray tan spray', 10, 4, 4, cannot_carry=True),
            Item('beach towel', 9, 10, 10, cannot_carry=True),
            Item('travel sunscreen', 9, 4, 4),
            Item('swimsuit', 20, 1, 1),
        ]

        pairs = [
            (items[1], items[3]),
            (items[2], items[4]),
        ]

        v_cap = 0
        w_cap = 10

        result = dp_choose_packing(items, pairs, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, pairs, v_cap, w_cap)
        self.assertEqual(result[0], 19)

    def test_dp_choose_packing_carryon_greedy_not_optimal(self):
        # test case where greedy solution that puts items in carry-on till full isn't optimal
        items = [
            Item('baked sweet potato', 1, 3, 10),
            Item('block of cheese', 1, 3, 3, cannot_carry=True),
            Item('extra socks', 10, 6, 20), # greedy would put 1st two in carry-on, but optimal is to put this one in carry-on
            Item('laptop', 5, 5, 10, cannot_carry=True),
            Item('laptop charger', 5, 5, 4, cannot_carry=True),
            Item('6-7 packs of gum', 2, 5, 4),
        ]

        pairs = [
            (items[3], items[4]),
        ]

        v_cap = 6  # Carry-on volume capacity
        w_cap = 15  # Checked bag weight capacity

        result = dp_choose_packing(items, pairs, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, pairs, v_cap, w_cap)
        self.assertEqual(result[0], 20)

    def test_dp_choose_packing_checked_greedy_not_optimal(self):
        # test case where greedily filling the checked first is not optimal
        items = [
            Item('sandals', 1, 3, 5),
            Item("scented hand lotion", 1, 3, 5),
            Item('winter coat', 10, 6, 10), # greedy would put 1st two in checked, but optimal is to put this one in checked
            Item('career fair t-shirts', 5, 5, 10),
        ]

        pairs = []

        v_cap = 2
        w_cap = 10

        result = dp_choose_packing(items, pairs, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, pairs, v_cap, w_cap)
        self.assertEqual(result[0], 10)

    def test_dp_choose_packing_constraints(self):
        # test case with significant constraints
        items = [
            Item('tablet', 1, 3, 5),
            Item('extra shoes', 1, 3, 5),
            Item('camera', 10, 6, 10, cannot_check=True),
            Item('portable battery', 5, 5, 10, cannot_carry=True),
            Item('tablet stylus', 5, 5, 4, cannot_carry=True),
        ]

        pairs = [
            (items[0], items[4]),
            (items[2], items[3])
        ]

        v_cap = 5
        w_cap = 10

        result = dp_choose_packing(items, pairs, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, pairs, v_cap, w_cap)
        self.assertEqual(result[0], 7)

    # dp-specific test case, will time out for brute-force version
    def test_dp_choose_packing_big(self):
        # large test case where brute-force solution would hang for a long time
        random.seed(21)
        # 20 random items
        items = [Item(f'Item {i}', random.randint(5, 20), random.randint(2, 10), random.randint(2, 10)) for i in range(20)]
        pairs = [
            (items[6], items[7]),
            (items[4], items[16]),
        ]
        v_cap = 50
        w_cap = 50

        result = dp_choose_packing(items, pairs, v_cap, w_cap)
        check_packing_returntype(result)
        check_valid_packing(*result, items, pairs, v_cap, w_cap)
        self.assertEqual(result[0], 267)

point_values = {
    "test_dp_choose_packing_all_fit_no_pairs" : 0.75,
    "test_dp_choose_packing_not_all_fit_no_pairs" : 0.75,
    "test_dp_choose_packing_all_fit_with_pairs" : 0.5,
    "test_dp_choose_packing_first_not_optimal" : 0.5,
    "test_dp_choose_packing_carryon_greedy_not_optimal" : 0.5,
    "test_dp_choose_packing_checked_greedy_not_optimal" : 0.5,
    "test_dp_choose_packing_constraints" : 0.5,
    "test_dp_choose_packing_big" : 2,
}
# Dictionary mapping function names from the above TestCase class to
# messages you'd like the student to see if their code throws an error.
error_messages = {
    "test_dp_choose_packing_all_fit_no_pairs" : "Your function dp_choose_packing() produced an error.",
    "test_dp_choose_packing_not_all_fit_no_pairs" : "Your function dp_choose_packing() produced an error.",
    "test_dp_choose_packing_all_fit_with_pairs" : "Your function dp_choose_packing() produced an error.",
    "test_dp_choose_packing_first_not_optimal" : "Your function dp_choose_packing() produced an error.",
    "test_dp_choose_packing_carryon_greedy_not_optimal" : "Your function dp_choose_packing() produced an error.",
    "test_dp_choose_packing_checked_greedy_not_optimal" : "Your function dp_choose_packing() produced an error.",
    "test_dp_choose_packing_constraints" : "Your function dp_choose_packing() produced an error.",
    "test_dp_choose_packing_big" : "Your function dp_choose_packing() produced an error.",
    }
# Dictionary mapping function names from the above TestCase class to
# messages you'd like the student to see if the test fails.
failure_messages = {
    "test_dp_choose_packing_all_fit_no_pairs" : "Your function dp_choose_packing() produced incorrect output.",
    "test_dp_choose_packing_not_all_fit_no_pairs" : "Your function dp_choose_packing() produced incorrect output.",
    "test_dp_choose_packing_all_fit_with_pairs" : "Your function dp_choose_packing() produced incorrect output.",
    "test_dp_choose_packing_first_not_optimal" : "Your function dp_choose_packing() produced incorrect output.",
    "test_dp_choose_packing_carryon_greedy_not_optimal" : "Your function dp_choose_packing() produced incorrect output.",
    "test_dp_choose_packing_checked_greedy_not_optimal" : "Your function dp_choose_packing() produced incorrect output.",
    "test_dp_choose_packing_constraints" : "Your function dp_choose_packing() produced incorrect output.",
    "test_dp_choose_packing_big" : "Your function dp_choose_packing() produced incorrect output.",
}

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPS1)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
