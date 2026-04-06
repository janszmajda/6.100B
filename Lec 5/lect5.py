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
# Produce plot related to quiz 1 question about using ternary search
############################################################

def find_landing(boat_speed, march_speed, dist_south, dist_east):
    times = []
    incr = 1
    miles_south = 0
    change_point = 64

    while miles_south <= dist_south:
        if miles_south > change_point:
            boat_speed *= 1.006
        dist_marched = ((dist_south-miles_south)**2+dist_east**2)**0.5
        tot_time = miles_south/boat_speed + dist_marched/march_speed
        times.append(tot_time)
        miles_south += incr
    plt.plot(times,
          label = f'boat speeds = {boat_speed:.2f}, {boat_speed*1.006:.2f} '
                  f'after mile {change_point}\nmarch_speed = {march_speed}')
    plt.title('Landing vs Time')
    plt.xlabel('Distance Down River')
    plt.ylabel('Total Time (hours)')
    plt.legend()

# find_landing(1.5, 1, 100, 50)
# sys.exit()

############################################################
# ROULETTE from last lecture
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
    winnings = []
    for _ in range(num_trials):
        winnings.append(play_roulette(game, num_spins, 2, bet))
    amt = sum(winnings)/(len(winnings)*num_spins)
    ave_return = f"{'-' if amt < 0 else ''}${abs(amt):.2f}"
    print(f'Ave. return per ${bet} bet over {num_spins:,} '
          f'spins of {game_name} = {ave_return}')
    plt.hist(winnings, bins = 20)
    plt.title(f'Total Return on Betting $10 {num_spins:,} Times\n'
              f'({num_trials:,} trials of {game_name})')
    plt.xlabel('Return ($)')
    plt.ylabel('Number of Times')
    plt.semilogy()

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


############################################################
# Code for this lecture
############################################################

def sim_game(game, spin_list, num_trials):
    means, stds = [], []
    for num_spins in spin_list:
        print(f'Simulate {num_trials:,} trials of {num_spins:,}'
              f' spins each of {game.__name__}')
        pocket_returns = [play_roulette(game(), num_spins, pocket=2, bet=1)
                          for _ in range(num_trials)]
        mean = sum(pocket_returns)/len(pocket_returns)
        means.append(mean)
        std_dev = np.std(pocket_returns)
        stds.append(std_dev)
        print(f'Estimated return = {mean:,}'
              f' ±{round(1.96*std_dev):,} with 95% confidence')
    return means, stds

def sim_game(game, spin_list, num_trials, std_or_cv = 'std'):
    means, stds = [], []
    for num_spins in spin_list:
        print(f'Simulate {num_trials:,} trials of {num_spins:,}'
              f' spins each of {game.__name__}')
        pocket_returns = [play_roulette(game(), num_spins, pocket=2, bet=1)
                          for _ in range(num_trials)]
        mean = sum(pocket_returns)/len(pocket_returns)
        means.append(mean)
        std_dev = np.std(pocket_returns)
        stds.append(std_dev)
        if std_or_cv == 'std':
            print(f'Estimated return = {mean:,}'
                  f' ±{round(1.96*std_dev):,} with 95% confidence')
        else:
            print(f'Estimated return = {mean:,}'
                  f' ±{round(1.96*std_dev):,} with 95% confidence. '
                  f'CV = {std_dev/abs(mean):.2f}')
    return means, stds

# Compare games

std_or_cv = 'cv'
spin_list = [10**p for p in range(4, 7)]
game_list = (Fair_roulette, Eu_roulette)
actual_returns = {Fair_roulette: 0.0, Eu_roulette: -0.02703}
num_trials = 20
for g in game_list:
    random.seed(1)
    means, stds = sim_game(g, spin_list, num_trials, std_or_cv)
    print()
    for i in range(len(spin_list)):
        means[i] = means[i]/spin_list[i]
        stds[i] = stds[i]/spin_list[i]

plt.errorbar(spin_list, means, yerr = stds, fmt = 'o')
plt.xlabel('Number of Spins')
plt.ylabel('Mean Return/Spin ± Std')
plt.axhline(y = -0.027, color = 'r')
plt.semilogx()
plt.title('Mean Return of European Roulette')

# sys.exit()


############################################################
# BIRTHDAY PROBLEM
############################################################

def true_prob(num_people):
    # Assumes each birthdate is equally probable
    unique_possibilities = math.factorial(366) // math.factorial(366 - num_people)
    total_possibilities = 366**num_people
    return 1 - unique_possibilities / total_possibilities

# for i in range(1, 9):
#     n = 2**i
#     print(f"Actual prob. of a shared birthday "
#           f"with {n} people = {true_prob(n):.3f}")

# sys.exit()

def same_date(num_people, num_same, birthday_probs):
    """num_people and num_same are positive ints
       possible dates is a non_empty sequence of ints between
       1 and 366.
       returns true if num_same people out of num_people share
       a birthday chosen at random from possible_dates"""
    possible_dates = range(366)
    birthday_counts = [0] * 366
    birth_dates = np.random.choice(possible_dates, size = num_people,
                                      p = birthday_probs)
    for birth_date in range(len(birth_dates)):
        birthday_counts[birth_dates[birth_date]] += 1
    return max(birthday_counts) >= num_same

def birthday_prob(num_people, num_same, birthday_probs, num_trials):
    num_hits = 0
    for _ in range(num_trials):
        if same_date(num_people, num_same, birthday_probs):
            num_hits += 1
    return num_hits / num_trials

# np.random.seed(0)

# birthday_probs = np.array([1 for _ in range(366)])/366
# num_same = 2
# for i in range(1, 9):
#     num_people = 2**i
#     print(f'Est. prob. of {num_same} people sharing a birthday, '
#       f'with {num_people} people: '
#       f'{birthday_prob(num_people, num_same, birthday_probs, 10_000):.3f}')

# sys.exit()


def get_bdays():
    infile = open('births.csv')
    infile.readline() # discard first line
    num_births = [int(line.split(',')[2][:-1]) for line in infile]
    birthday_probs = np.array(num_births)/sum(num_births)
    return birthday_probs

# np.random.seed(1)
# trials = 100_000
# birthday_probs = get_bdays()
# uniform_probs = np.array([1 for _ in range(366)])/366
# num_same = 4
# for i in range(4, 9):
#     num_people = 2**i
#     print(f"For population {num_people}, probability of {num_same} "
#           f"shared birthdays:")
#     print(f"  Est. for actual distribution: "
#           f"{birthday_prob(num_people, num_same, birthday_probs, trials):.5f}")
#     print(f"  Est. for uniform distribution: "
#           f"{birthday_prob(num_people, num_same, uniform_probs, trials):.5f}")

# sys.exit()

# Used to produce plot shown in lecture
# Plot distribution of dates in birthdates file
def plot_bday_dist():
    infile = open('births.csv')
    infile.readline() # discard first line
    num_births = [int(line.split(',')[2][:-1]) for line in infile]
    d = {i+1: num_births[i] for i in range(len(num_births))}
    vals = [d[k] for k in d.keys()]
    plt.plot(vals, 'bo')
    plt.xlim(-10, plt.xlim()[1])
    plt.ylim(0, plt.ylim()[1])
    plt.xlabel("Day of Year")
    plt.ylabel("num_ber of Births")
    mean = f"Mean = {int(sum(vals)/len(vals))}"
    std = f"Std = {int(np.std(vals))}"
    plt.title(f"Frequency of Birthdates\n{mean}, {std}")

# plot_bday_dist()
# plt.show()


############################################################
# Random Walks
############################################################

class Location(object):
    def __init__(self, x, y):
        """x and y are numbers"""
        self.x = x
        self.y = y

    def move(self, delta_x, delta_y):
        """deltaX and deltaY are numbers"""
        return Location(self.x + delta_x,
                        self.y + delta_y)
    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def dist_from(self, other):
        x_dist = self.x - other.get_x()
        y_dist = self.y - other.get_y()
        return (x_dist**2 + y_dist**2)**0.5

    def __str__(self):
        return '<' + str(self.x) + ', ' + str(self.y) + '>'

class Field(object):
    def __init__(self):
        self.drunks = {}

    def add_drunk(self, drunk, loc):
        if drunk in self.drunks:
            raise ValueError('Duplicate drunk')
        else:
            self.drunks[drunk] = loc

    def get_loc(self, drunk):
        if drunk not in self.drunks:
            raise ValueError('Drunk not in field')
        return self.drunks[drunk]

    def move_drunk(self, drunk):
        if drunk not in self.drunks:
            raise ValueError('Drunk not in field')
        x_dist, y_dist = drunk.take_step()
        # use move method of Location to set new location
        self.drunks[drunk] =\
            self.drunks[drunk].move(x_dist, y_dist)

class Drunk(object):
    def __init__(self, name = None):
        """Assumes name is a str"""
        self.name = name

    def __str__(self):
        if self != None:
            return self.name
        return 'Anonymous'

class Usual_drunk(Drunk):
    def take_step(self):
        step_choices = [(0,1), (0,-1),
                       (1, 0), (-1, 0)]
        return random.choice(step_choices)

class Masochist_drunk(Drunk):
    def take_step(self):
        step_choices = [(0.0,1.1), (0.0,-0.9),
                       (1.0, 0.0), (-1.0, 0.0)]
        return random.choice(step_choices)

class Liberal_drunk(Drunk):
    def take_step(self):
        step_choices = [(0.0,1.0), (0.0,-1.0),
                       (0.9, 0.0), (-1.1, 0.0)]
        return random.choice(step_choices)

class Conserative_drunk(Drunk):
    def take_step(self):
        step_choices = [(0.0,1.0), (0.0,-1.0),
                       (1.1, 0.0), (-0.9, 0.0)]
        return random.choice(step_choices)

class Liberal_masochist_drunk(Masochist_drunk):
    def take_step(self):
        if random.choice([True, False]):
            step_choices = [(0.0,1.0), (0.0,-1.0),
                           (0.9, 0.0), (-1.1, 0.0)]
            return random.choice(step_choices)
        else:
            return Masochist_drunk.take_step(self)

def walk(f, d, num_steps):
    """Assumes: f a Field, d a Drunk in f,
                and num_steps an int >= 0.
       Moves d num_steps times, and returns the distance between the
       final location and the location at the start of the walk."""
    start = f.get_loc(d)
    for s in range(num_steps):
        f.move_drunk(d)
    return start.dist_from(f.get_loc(d))

def sim_walks(num_steps, num_trials, d_class):
    """Assumes num_steps an int >= 0, num_trials an int > 0,
         d_class a subclass of Drunk
       Simulates num_trials walks of num_steps steps each.
       Returns a list of the final distances for each trial"""
    Homer = d_class('Homer')
    origin = Location(0, 0)
    distances = []
    for t in range(num_trials):
        f = Field()
        f.add_drunk(Homer, origin)
        distances.append(round(walk(f, Homer, num_trials), 1))    # caution, an error: num_steps
    return distances

def drunk_test(walk_lengths, num_trials, d_class):
    """Assumes walk_lengths a sequence of ints >= 0
         num_trials an int > 0, d_class a subclass of Drunk
       For each number of steps in walk_lengths, runs
       sim_walks with num_trials walks and prints results"""
    for num_steps in walk_lengths:
        distances = sim_walks(num_steps, num_trials, d_class)
        print(f'{d_class.__name__} random walk of {num_steps:,} steps')
        print(f' Mean = {sum(distances)/len(distances):.2f}')
        print(f' Max = {max(distances):.2f}, Min = {min(distances):.2f}')

random.seed(0)
drunk_test((10, 100, 1000, 10000), 100, Usual_drunk)
drunk_test((0, 1, 2), 100, Usual_drunk)

def plot_drunk_test(walk_lengths, num_trials, d_class):
    """Assumes walk_lengths a sequence of ints >= 0
         num_trials an int > 0, d_class a subclass of Drunk
       Plots the average distance for each walk length and
         the sqrt of each walk length"""
    means = []
    for wl in walk_lengths:
        distances = (sim_walks(wl, num_trials, d_class))
        means.append(sum(distances)/len(distances))
    plt.plot(walk_lengths, means, label = 'Distance')
    roots = [wl**0.5 for wl in walk_lengths]
    plt.plot(walk_lengths, roots, '--', label = 'Sqrt of steps')
    plt.semilogy()
    plt.semilogx()
    plt.xlabel('Steps Taken')
    plt.ylabel('Distance from Origin')
    plt.title('Mean Distance from Origin\n100 Trials')
    plt.legend()

# walk_lengths = []
# for i in range(1,6):
#     walk_lengths.append(10**i)
# plot_drunk_test(walk_lengths, 100, Continuous_drunk)

### sim_all that just outputs values
def sim_all(drunk_kinds, walk_lengths, num_trials):
    for d_class in drunk_kinds:
        random.seed(1)
        drunk_test(walk_lengths, num_trials, d_class)

sim_all((Usual_drunk, Masochist_drunk), (1000, 10000), 100)

class Random_style_generator:
    def __init__(self):
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.lines = ['-', ':', '--', '-.']
        self.used_styles = set()

    def get_random_plot_style(self):
        """
        Returns a random plot style (color and marker combination).
        Avoids repeating the same style.
        """
        available_styles = [(color, line) for color in self.colors
                            for line in self.lines
                            if (color, line) not in self.used_styles]
        if not available_styles:
            # All styles have been used, reset the set
            self.used_styles.clear()
            available_styles = [(color, line)
                                for color in self.colors
                                for line in self.lines]
        random_color, random_line = random.choice(available_styles)
        self.used_styles.add((random_color, random_line))
        return random_color + random_line


def sim_drunk(num_trials, d_class, walk_lengths):
    mean_distances = []
    for num_steps in walk_lengths:
        print('Starting simulation of', num_steps, 'steps')
        trials = sim_walks(num_steps, num_trials, d_class)
        mean = sum(trials)/len(trials)
        mean_distances.append(mean)
    return mean_distances

## sim_all that outputs a visualization
def sim_all(drunk_kinds, walk_lengths, num_trials):
    style_choice = Random_style_generator()
    for d_class in drunk_kinds:
        cur_style = style_choice.get_random_plot_style()
        print('Starting simulation of', d_class.__name__)
        means = sim_drunk(num_trials, d_class, walk_lengths)
        plt.plot(walk_lengths, means, cur_style,
                 label = d_class.__name__)
    plt.title(f'Mean Distance from Origin ({num_trials} trials)')
    plt.xlabel('Number of Steps')
    plt.ylabel('Distance from Origin')
    plt.legend(loc = 'best')


# random.seed(0)
# num_steps = (10, 100, 1000, 10000)
# sim_all((Usual_drunk, Masochist_drunk, Liberal_drunk,
#          Liberal_masochist_drunk), num_steps, 100)
# plt.legend(loc = 'best')

def sim_final_locs(num_steps, num_trials, d_class):
    locs = []
    d = d_class()
    for t in range(num_trials):
        f = Field()
        f.add_drunk(d, Location(0, 0))
        for s in range(num_steps):
            f.move_drunk(d)
        locs.append(f.get_loc(d))
    return locs

def plot_locs(drunk_kinds, num_steps, num_trials):
    style_choice = Random_style_generator()
    for d_class in drunk_kinds:
        locs = sim_final_locs(num_steps, num_trials, d_class)
        x_vals, y_vals = [], []
        for loc in locs:
            x_vals.append(loc.get_x())
            y_vals.append(loc.get_y())
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        mean_x = round(sum(x_vals)/len(x_vals))
        mean_y = round(sum(y_vals)/len(y_vals))
        abs_mean_x = round(sum(abs(x_vals))/len(x_vals))
        abs_mean_y = round(sum(abs(y_vals))/len(y_vals))
        cur_style = style_choice.get_random_plot_style()
        plt.scatter(x_vals, y_vals, color = cur_style[0],
                     label = d_class.__name__ + ':' +
                     ' mean abs dist = <'
                     + str(abs_mean_x) + ', ' + str(abs_mean_y) + '>\n'
                     'mean dist = <'+str(mean_x) + ', '+str(mean_y)+'>')
    plt.title('Location at End of Walks (' + str(num_steps) + ' steps)')
    plt.ylim(-1000, 1000)
    plt.xlim(-1000, 1000)
    plt.xlabel('Steps East/West of Origin')
    plt.ylabel('Steps North/South of Origin')
    plt.legend(loc = 'lower center', fontsize = 'large')

# random.seed(1)
# plot_locs((Usual_drunk, Masochist_drunk, Liberal_masochist_drunk), 10000, 100)
