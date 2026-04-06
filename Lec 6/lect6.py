import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys

#set line width
plt.rcParams['lines.linewidth'] = 4
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
# Code from last lecture
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

class Conservative_drunk(Drunk):
    def take_step(self):
        step_choices = [(0.0,1.0), (0.0,-1.0),
                       (1.1, 0.0), (-0.9, 0.0)]
        return random.choice(step_choices)

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
        distances.append(round(walk(f, Homer, num_steps), 1))
    return distances

def drunk_test(walk_lengths, num_trials, d_class):
    """Assumes walk_lengths a sequence of ints >= 0
         num_trials an int > 0, d_class a subclass of Drunk
       For each number of steps in walk_lengths, runs
       sim_walks with num_trials walks and prints results"""
    for num_steps in walk_lengths:
        distances = sim_walks(num_steps, num_trials, d_class)
        print(d_class.__name__, 'random walk of', num_steps, 'steps')
        print(' Mean =', round(sum(distances)/len(distances), 4))
        print(' Max =', max(distances), 'Min =', min(distances))

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
    # # Apply contstant correction where known
    # if d_class == Usual_drunk:
    #     using_c2 = [(np.pi**0.5/2)*wl**0.5 for wl in walk_lengths]
    #     plt.plot(walk_lengths, using_c2, 'X',
    #              label = r'$\sqrt{\frac{\pi}{2}}*\sqrt{|steps|}$')
    # if d_class == Linear_drunk:
    #     using_c1 = [(2/np.pi)**0.5*(wl**0.5) for wl in walk_lengths]
    #     plt.plot(walk_lengths, using_c1, 'X',
    #              label = r'$\sqrt{2/\pi}*\sqrt{|steps|}$')
    plt.semilogy()
    plt.semilogx()
    plt.xlabel('Steps Taken')
    plt.ylabel('Distance from Origin')
    plt.title(f'{d_class.__name__} Mean Distance from Origin\n{num_trials:,} Trials')
    plt.legend()

# walk_lengths = []
# for i in range(1,6):
#     walk_lengths.append(10**i)
# plot_drunk_test(walk_lengths, 500, Usual_drunk)
# sys.exit()

## New code

# sim_all that just outputs values
def sim_all(drunk_kinds, walk_lengths, num_trials):
    for d_class in drunk_kinds:
        random.seed(1)
        drunk_test(walk_lengths, num_trials, d_class)

# sim_all((Usual_drunk, Masochist_drunk), (1000, 10_000), 100)
# sys.exit()

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
    mean_dists = []
    for num_steps in walk_lengths:
        print(f'Starting simulation of {num_steps:,} steps')
        trials = sim_walks(num_steps, num_trials, d_class)
        mean = sum(trials)/len(trials)
        mean_dists.append(mean)
    return mean_dists

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
#          Conservative_drunk), num_steps, 100)
# plt.legend(loc = 'best')
# plt.show()
# sys.exit()

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
        cur_style = style_choice.get_random_plot_style()
        plt.scatter(x_vals, y_vals, color = cur_style[0],
                    label = d_class.__name__)
    plt.title(f'Location at End of Walks ({num_steps:,}) steps)')
    plt.ylim(-1000, 1000)
    plt.xlim(-1000, 1000)
    plt.xlabel('Steps East/West of Origin')
    plt.ylabel('Steps North/South of Origin')
    plt.legend(loc = 'lower center', fontsize = 'large')

# random.seed(0)
# plot_locs((Usual_drunk, Liberal_drunk, Conservative_drunk), 10000, 100)
# sys.exit()

## Simulations related to gasses

class Inertial_drunk(Drunk):
    def __init__(self, name):
       Drunk.__init__(self, name)
       self.last_step = None
    def reset_last_step(self):
        self.last_step = None
    def take_step(self, new_dir = False):
        if self.last_step == None:
            delta_func = (lambda: random.random()
                          if random.random() < 0.5 else -random.random())
            delta_x = delta_func()
            delta_y = delta_func()
            self.last_step = (delta_x, delta_y)
            return (delta_x, delta_y)
        else:
            return self.last_step

## simulate with multiple drunks, don't allow collisions
## include boundaries on size of field

class Field_multi_opt_particles(object):
    """ Optimized for large number of particles"""
    def __init__(self, x_lim, y_lim):
        self.drunks = {}
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.wall_hits, self.collisions = 0, 0
        self.drunks_by_loc = {(x, y): []
                          for x in range(0, x_lim)
                          for y in range(0, y_lim)}

    def add_drunk(self, drunk, loc):
        if drunk in self.drunks:
            raise ValueError('Duplicate drunk')
        else:
            self.drunks[drunk] = loc
            self.drunks_by_loc[(loc.get_x(), loc.get_y())].append(drunk)

    def get_loc(self, drunk):
        if drunk not in self.drunks:
            raise ValueError('Drunk not in field')
        return self.drunks[drunk]

    def get_lims(self):
        return (self.x_lim, self.y_lim)

    def get_wall_hits(self):
        return self.wall_hits

    def get_collisions(self):
        return self.collisions

    def move_drunk(self, drunk):
        def find_relevant_cells(d_col, d_row):
            # Find relevant columns
            relevant_cols = [d_col]
            if d_col < self.x_lim - 1: # not at right edge
                relevant_cols.append(d_col + 1)
            if d_col > 0: # not at left edge
                relevant_cols.append(d_col - 1)
            relevant_cells = [(x, d_row) for x in relevant_cols]
            if d_row < self.y_lim - 1: # not at top edge
                for d_col in relevant_cols:
                    relevant_cells.append((d_col, d_row + 1))
            if d_row > 0: # not at bottom edge
                for d_col in relevant_cols:
                    relevant_cells.append((d_col, d_row - 1))
            return relevant_cells
        if drunk not in self.drunks:
            raise ValueError('Drunk not in field')
        old_x = int(self.drunks[drunk].get_x())
        old_y = int(self.drunks[drunk].get_y())
        # use move method of Location to get new location within field
        # respect edges of field
        x_dist, y_dist = drunk.take_step()
        new_loc = self.drunks[drunk].move(x_dist, y_dist)
        x_val, y_val = new_loc.get_x(), new_loc.get_y()
        d_col = int(x_val)
        d_row = int(y_val)
        # Check if would hit a wall
        if new_loc.get_x() < 0 or new_loc.get_x() > self.x_lim - 1:
            self.wall_hits += 1
            drunk.reset_last_step() # So particle will bounce on next move
        elif new_loc.get_y() < 0 or new_loc.get_y() > self.y_lim - 1:
            self.wall_hits += 1
            drunk.reset_last_step() # So particle will bounce on next move
        else: #check if would collide with another molecule
            # Limit area to search for collisions
            relevant_cells = find_relevant_cells(d_col, d_row)
            possible_neighbors = []
            for cell in relevant_cells:
                for d in self.drunks_by_loc[cell]:
                    if d != drunk:
                        possible_neighbors.append(d)
            for d in possible_neighbors:
                # check that not too close to other drunks
                if d != drunk and self.drunks[d].dist_from(new_loc) < 1.0:
                    d.reset_last_step() # So next move will be random
                    self.collisions += 1
                    return
            self.drunks[drunk] = new_loc
            self.drunks_by_loc[(old_x, old_y)].remove(drunk)
            self.drunks_by_loc[int((new_loc.get_x())),
                                   int(new_loc.get_y())].append(drunk)

def walk_multi(f, ds, num_steps, starts):
    """Assumes: f a Field, ds a dict of Drunks in f,
          num_steps an int >= 0,
          starts a dict of starting locations
       Moves each d in ds num_steps times, and returns the
       average distance between the
       final location and the location at the start of the walk."""
    for s in range(num_steps):
        for d in ds:
            f.move_drunk(d)
    dists = []
    for d in ds:
        start = starts[d]
        dist = start.dist_from(f.get_loc(d))
        dists.append(dist)
    return dists

def sim_walks_multi(num_steps, num_trials, d_class, num_drunks,
                    boundaries, verbose = False):
    """Assumes num_steps an int >= 0, num_trials an int > 0,
         d_class a subclass of Drunk, num_drunks an int,
         boundaries an int
       Simulates num_trials walks of num_steps steps each.
       Returns a list of the final mean, max, and min distances
         for each trial"""
    starts = {}
    range_of_choices = [i for i in range(0, boundaries)]
    mean_dists, max_dists, min_dists, wall_hits, cols = [], [], [], [], []
    for t in range(num_trials):
        f = Field_multi_opt_particles(boundaries, boundaries)
        for i in range(num_drunks):
            Homer = d_class('Homer' + str(i))
            start = Location(random.choice(range_of_choices),
                             random.choice(range_of_choices))
            f.add_drunk(Homer, start)
            starts[Homer] = start
        distances = walk_multi(f, f.drunks, num_steps, starts)
        mean_dists.append(sum(distances)/len(distances))
        max_dists.append(max(distances))
        min_dists.append(min(distances))
        wall_hits.append(f.get_wall_hits())
        cols.append(f.get_collisions())
        if verbose:
            print(f'Mean distance for trial {t} = {mean_dists[-1]}')
    return mean_dists, max_dists, min_dists, wall_hits, cols

def drunk_test_multi(walk_lengths, num_trials, d_class, num_particles,
                     boundaries, verbose = False):
    """Assumes walk_lengths a sequence of ints >= 0
         num_trials an int > 0, d_class a subclass of Drunk
       For each number of steps in walk_lengths, runs
       sim_walks with num_trials walks and prints results"""
    for l in walk_lengths:
        for num in num_particles:
            for d in boundaries:
                print(f'particles = {num}, size = {d:,}X{d:,}, steps = {l:,}')
                mean_dists, max_dists, min_dists, wall_hits, cols =\
                            sim_walks_multi(l, num_trials, d_class, num, d,
                                            verbose = verbose)
                max_d = round(max(max_dists))
                min_d = round(min(min_dists))
                mean_d = sum(mean_dists)/len(mean_dists)
                mean_wh = sum(wall_hits)/len(wall_hits)
                mean_col = sum(cols)/len(cols)
                print(f' Distance: Max = {max_d:,}, Min = {min_d:,},',
                      f'Mean = {round(mean_d):,}')
                print(f' Mean wall hits = {round(mean_wh):,}')
                if mean_col != 0:
                    print(f' Mean collisions = {round(mean_col):,}')
    return (mean_d, mean_wh, mean_col)

# # Vary field size
# random.seed(1)
# num_particles = (1,)
# sizes = (10, 20, 50, 100, 1000)
# lengths = (50000,)
# num_trials = 50
# drunk_test_multi(lengths, num_trials, Inertial_drunk, num_particles, sizes)
# sys.exit()

# # Vary walk length (can be thought of a velocity, which is related to temp.)
# random.seed(1)
# num_particles = (1,)
# sizes = (100,)
# lengths = (50, 75, 100, 125, 150, 175)
# lengths = [2**p for p in range(5, 11)]
# num_trials = 50
# drunk_test_multi(lengths, num_trials, Inertial_drunk, num_particles, sizes)

# sys.exit()

# # Vary number of particles per trial, corresponds to density
# random.seed(1)
# sizes = (30,)
# lengths = (200,)
# num_trials = 400
# num_particles = [2,16,64,128,512,1024]
# wall_hits, collisions = [], []
# for n in num_particles:
#     mean_d, mean_wh, mean_col =\
#          drunk_test_multi(lengths, num_trials, Inertial_drunk, (n,),
#                           sizes, verbose = False)
#     wall_hits.append(mean_wh)
#     collisions.append(mean_col/n/lengths[0])

# print(collisions)

# # Plot results from simulation varying number of particles

# plt.plot(num_particles, wall_hits,'o-', label = 'Data points')
# plt.title('Pressure vs. Number of Particles\n'
#           f'(Size = {sizes[0]:,}X{sizes[0]:,})')
# plt.xlabel('Number of Particles')
# plt.ylabel('Number of Wall Hits')

# # Fit a model to predict total number of wall hits
# model = np.polyfit(num_particles, wall_hits, 1)
# plt.plot(num_particles, np.polyval(model, num_particles), 'k',
#          label = 'Linear model')

# plt.legend()

# plt.figure()
# plt.plot(num_particles, collisions,'o-')
# plt.title('Particle Collisions vs. Number of Particles\n'
#           f'(Size = {sizes[0]:,}X{sizes[0]:,})')
# plt.xlabel('Number of Particles')
# plt.ylabel('Collisions/Step/Particle')

# sys.exit()

############################################################
# ESTIMATING PI
############################################################


def throw_needles(num_needles):
    in_circle = 0
    for _ in range(num_needles):
        x = random.random()
        y = random.random()
        if (x*x + y*y)**0.5 <= 1:
            in_circle += 1
    # Counting needles in one quadrant only, so multiply by 4
    return 4 * in_circle / num_needles

def throw_needles(num_needles):
    x = np.random.random(num_needles)
    y = np.random.random(num_needles)
    in_circle = np.sum(x*x + y*y <= 1)
    # Counting needles in one quadrant only, so multiply by 4
    return 4 * in_circle / num_needles

def get_estimate(num_needles, num_trials):
    estimates = [throw_needles(num_needles) for _ in range(num_trials)]
    cur_est = sum(estimates)/len(estimates)
    std_dev = np.std(estimates)
    return (cur_est, std_dev)

# Statistically correct version
def get_estimate(num_needles, num_trials):
    estimates = [throw_needles(num_needles) for _ in range(num_trials)]
    cur_est = estimates[0]
    std_dev = np.std(estimates)
    return (cur_est, std_dev)

# np.random.seed(0)
# for p in range(1, 6):
#     n = 10**p
#     est, dev = get_estimate(n, 100)
#     print(f'Trial with {n:,} needles: π = {est:.5f}, std = {dev:.5f}')
#     print(f'   Error wrt to np.pi = {abs(np.pi - est):.5f}')

# sys.exit()

# def estimate_pi_CI(precision, num_trials):
#     num_needles = 100
#     std_dev = precision
#     while 1.96 * std_dev >= precision:
#         cur_est, std_dev = get_estimate(num_needles, num_trials)
#         print(f"Needles = {num_needles:,}, "
#               f"π = {cur_est:.5g}, std dev = {std_dev:.5g}")
#         num_needles *= 2
#     return cur_est


def estimate_pi_CI_plot(precision, num_trials):
    num_needles = 100_000
    std_dev = precision
    needles = []
    estimates = []
    error_bars = []
    while 1.96 * std_dev >= precision:
        cur_est, std_dev = get_estimate(num_needles, num_trials)
        print(f'Needles = {num_needles:,}, '
              f'estimate = {cur_est:.5f}±{1.96*std_dev:.5f}')
        needles.append(num_needles)
        estimates.append(cur_est)
        error_bars.append(1.96 * std_dev)
        num_needles *= 2
    plt.xlim(90_000, num_needles)
    plt.semilogx()
    plt.title("Estimate of π, with increasing needles, and 95% confidence")
    plt.xlabel("Number of needles")
    plt.ylabel("π estimate")
    plt.errorbar(needles, estimates, yerr=error_bars)


# np.random.seed(0)
# estimate_pi_CI_plot(0.00025, 100)
# sys.exit()


############################################################
# INTEGRATION BY SAMPLING
############################################################

def integrate(fcn, minX, maxX, num_samples = 1_000_000):
    under_curve = 0
    for _ in range(num_samples):
        x = random.uniform(minX, maxX)
        y = fcn(x)
        under_curve += y
    return under_curve / num_samples * (maxX - minX) #finding average height of fxn and multiplying by x-range

def integrate_and_print(fcn, minX, maxX, num_samples = 1_000_000):
    result = integrate(fcn, minX, maxX, num_samples)
    print(f'Est. integral of {fcn.__name__} from {minX:.2f} ' +
          f'to {maxX:.2f} = {result:.4f}')

def print_diff(fcn, minX, maxX):
    diff = abs(integrate(fcn, minX, maxX) -
            scipy.integrate.quad(fcn, minX, maxX)[0])
    print(f'  Difference between estimate and scipy quadrature is {diff:.4f}')

random.seed(0)

# integrate_and_print(np.sin, 0, np.pi)
# print_diff(np.sin, 0, np.pi)
# integrate_and_print(np.sin, 0, 2*np.pi)
# print_diff(np.sin, 0, 2*np.pi)
# integrate_and_print(np.cos, 0, np.pi)
# print_diff(np.cos, 0, np.pi)
# sys.exit()

def funky(x):
    if x <= 1:
        return x
    elif x <= 2:
        return 1-x
    return x-2

# funky_vals = []
# x_vals = [0.01*i for i in range(0, 301)]
# for x in x_vals:
#     funky_vals.append(funky(x))
# plt.plot(x_vals, funky_vals)
# plt.title('Funky Function')

# integrate_and_print(funky, 0, 1)
# print_diff(funky, 0, 1)

# integrate_and_print(funky, 0, 3)
# print_diff(funky, 0, 3)
