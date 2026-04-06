import random
import math
import numpy as np
import matplotlib.pyplot as plt
import sys


#set line width
plt.rcParams['lines.linewidth'] = 2
#set font size for titles
plt.rcParams['axes.titlesize'] = 16
#set font size for labels on axes
plt.rcParams['axes.labelsize'] = 16
#set size of numbers on x-axis
plt.rcParams['xtick.labelsize'] = 12
#set size of numbers on y-axis
plt.rcParams['ytick.labelsize'] = 12
#set size of ticks on x-axis
plt.rcParams['xtick.major.size'] = 7
#set size of ticks on y-axis
plt.rcParams['ytick.major.size'] = 7
#set size of markers, e.g., circles representing points
#set numpoints for legend
plt.rcParams['legend.numpoints'] = 1

############################################################
# ROLLING DICE
############################################################


def roll_die():
    """Return an int between 1 and 6"""
    return 2


def roll_die():
    """Return a random int between 1 and 6"""
    return random.choice([1, 2, 3, 4, 5, 6])


def test_roll(n):
    result = ''
    for _ in range(n):
        result += ' ' + str(roll_die())
    return result


# for _ in range(10):
#     print(test_roll(5))


def run_sim(goal, num_trials):
    total = 0
    for i in range(num_trials):
        if i != 0 and i % 100_000 == 0:
            print(f"Starting trial {i}")
        result = ''
        for _ in range(len(goal)):
            result += str(roll_die())
        if result == goal:
            total += 1
    print(f"Actual probability of {goal} = {1 / 6**len(goal):.8f}")
    print(f"Estimated Probability of {goal} = {total / num_trials:.8f}")

# random.seed(0)
# run_sim('11111', 1000)
# run_sim('11111', 1_000_000)

# sys.exit()


############################################################
# ROULETTE
############################################################


class Fair_roulette():

    def __init__(self):
        self.pockets = list(range(1, 37))
        self.ball = None
        self.pocket_odds = len(self.pockets) - 1

    def spin(self):
        self.ball = random.choice(self.pockets)

    def bet_pocket(self, pocket, amt):
        pocket = random.choice(self.pockets)
        if str(pocket) == str(self.ball):
            return amt * self.pocket_odds
        else:
            return -amt

    def __str__(self):
        return 'Fair roulette'

def play_roulette(game, num_spins, pocket, bet):
    tot_pocket = 0
    for _ in range(num_spins):
        game.spin()
        tot_pocket += game.bet_pocket(pocket, bet)
    return tot_pocket

def format_dollar(amount):
    return f"{'-' if amount < 0 else ''}${abs(amount):.2f}"

def test_roulette(game, num_spins, num_trials):
    game_name = type(game).__name__
    bet = 10
    winnings = 0
    for _ in range(num_trials):
        residual = play_roulette(game, num_spins, 2, bet)
        winnings += residual
        if num_trials < 11:
            print(f'Return on betting ${bet} on a pocket for {game_name} '
                  f'{num_spins} times = {format_dollar(residual)}\n')
    print(f'Average return = {format_dollar(winnings)}')

# random.seed(0)
# test_roulette(Fair_roulette(), 10, 3)
# sys.exit()


def test_roulette(game, num_spins, num_trials):
    game_name = type(game).__name__
    bet = 10
    winnings = []
    for _ in range(num_trials):
        winnings.append(play_roulette(game, num_spins, 2, bet))
    amt = sum(winnings)/(len(winnings)*num_spins)
    ave_return = f"{'-' if amt < 0 else ''}${abs(amt):.2f}"
    print(f'Ave. return per ${bet} bet over {num_spins:,} '
          f'spins of {game_name} = {ave_return}')
    plt.hist(winnings, bins = 20)
    plt.title(f'Total Return on Betting ${bet} {num_spins:,} Times\n'
              f'({num_trials:,} trials of {game_name})')
    plt.xlabel('Return ($)')
    plt.ylabel('Number of Times')
    plt.semilogy()

# random.seed(0)
# test_roulette(Fair_roulette(), 1000, 10_000)
# sys.exit()

############################################################
# REGRESSION TO MEAN
############################################################

# # Using Galton's regression coefficient to predict height
# L_James = 6*12 + 9
# S_James = 5*12 + 7
# Bronny = 6*12 + 4
# ave_man = 5*12 + 9
# ave_woman = 5*12 + 3.5
# parent_ave = (S_James + L_James)/2
# am_ave = (ave_woman + ave_man)/2
# diff = parent_ave - am_ave
# pred_Bronny = L_James - diff*(2/3)
# print(f'Predicted height = {pred_Bronny:.2f} inches')
# print(f'Actual height = {Bronny} inches')

# sys.exit()

def get_grade(student, num_questions):
    tot_right = 0
    for _ in range(num_questions):
        # student[i] is talent level. Higher is better
        if random.random() < student[1]:
            tot_right += 1
    return 10*(tot_right/num_questions)

def print_grades(grades):
    for g in grades:
        print(f'{g[0][0]}: {g[1]}')

random.seed(0)
num_students = 200
students = [(f'student{i}', random.choice((0.3, 0.4, 0.5, 0.6, 0.7)))
            for i in range(num_students)]
grades = []
for s in students:
    grades.append((s, get_grade(s, 4)))

# grades = sorted(grades, key = lambda x: x[1])
# best_grades = grades[-10:]
# worst_grades = grades[:10]
# best_grades = sorted(best_grades, key = lambda x: x[0])
# worst_grades = sorted(worst_grades, key = lambda x: x[0])
# best_students = [e[0] for e in best_grades]
# worst_students = [e[0] for e in worst_grades]

# print('Best Grades on Q1')
# print_grades(best_grades)
# print('Worst Grades on Q1')
# print_grades(worst_grades)
# best_mean = sum([e[1] for e in best_grades])/len(best_grades)
# worst_mean = sum([e[1] for e in worst_grades])/len(worst_grades)
# print(f'Mean for top 10 = {best_mean}, '
#       f'Mean for bottom 10 = {worst_mean}')

# new_grades = []
# for s in best_students:
#     new_grades.append((s, get_grade(s, 10)))
# print('Best Q1 Students on Q2')
# print_grades(new_grades)
# best_mean = sum([e[1] for e in new_grades])/len(new_grades)
# new_grades = []
# for s in worst_students:
#     new_grades.append((s, get_grade(s, 10)))
# print('Worst Q1 Students on Q2')
# print_grades(new_grades)
# worst_mean = sum([e[1] for e in new_grades])/len(new_grades)
# print(f'Mean on Q2 for top 10 on Q1 = {best_mean}, '
#       f'Mean for bottom 10 on Q1 = {worst_mean}')

# sys.exit()

class Eu_roulette(Fair_roulette):

    def __init__(self):
        super().__init__()
        self.pockets.append('0')

    def __str__(self):
        return "European roulette"


class Am_roulette(Eu_roulette):

    def __init__(self):
        super().__init__()
        self.pockets.append('00')

    def __str__(self):
        return "American roulette"

# for num_spins in (10, 100, 1000, 10_000):
#     for Game in (Fair_roulette, Eu_roulette, Am_roulette):
#         random.seed(0)
#         test_roulette(Game(), num_spins, 1000)
#         if Game != Am_roulette:
#             plt.figure()

# sys.exit()

def sim_double_strategy(game):
    max_down, num_spins = 0, 0
    bet = 10
    while True:
        game.spin()
        num_spins += 1
        result = game.bet_pocket(2, bet)
        if result < 0:
            max_down += result
            bet = bet*2
        else:
            return num_spins, max_down

random.seed(0)
worst_case_down, worst_case_spins = 0, 0
downs, spins = [], []
num_sims = 1000
for _ in range(num_sims):
    num_spins, max_down = sim_double_strategy(Am_roulette())
    spins.append(num_spins)
    downs.append(max_down)

print(f'Number of simulations = {num_sims:,}')
print('For betting a pocket in American Roulette')
print(f'Ave. spins before winning = {sum(spins)/len(spins)}')
print(f'Worst case spins before winning = {max(spins):,}')
print(f'Ave. amount down before winning = ${round(abs(sum(downs)/len(downs))):,}')
print(f'Worst case amount lost before winning = ${abs(min(downs)):,}')
