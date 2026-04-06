################################################################################
# 6.100B Spring 2026
# Problem Set 2
# Name: Jan Szmajda
# Collaborators: PyTutor
# Time: 8hrs

import random
from matplotlib.mlab import GaussianKDE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import wasserstein_distance
from mbta_helpers import History, produce_animation

# set line width
plt.rcParams['lines.linewidth'] = 4
# set font size for titles
plt.rcParams['axes.titlesize'] = 16
# set font size for labels on axes
plt.rcParams['axes.labelsize'] = 16
# set size of numbers on x-axis
plt.rcParams['xtick.labelsize'] = 12
# set size of numbers on y-axis
plt.rcParams['ytick.labelsize'] = 12
# set size of ticks on x-axis
plt.rcParams['xtick.major.size'] = 7
# set size of ticks on y-axis
plt.rcParams['ytick.major.size'] = 7
# set numpoints for legend
plt.rcParams['legend.numpoints'] = 1
# set marker size
plt.rcParams['lines.markersize'] = 10

############################################################
# Part 1: Track Simulations #
############################################################

class PerfectTrack(object):
    """
    A PerfectTrack is a discrete-time simulation of a one track circular train line
    in which all trains run at maximum speed at all times.
    """
    def __init__(self, max_speed):
        """
        Initializes a PerfectTrack in which all trains run at `max_speed`.

        Args:
            max_speed (float): the maximum speed of the trains in miles/min
        """
        stop_names = ['Alewife', 'Davis', 'Porter', 'Central', 'Kendall-MIT',
                      'Charles-MGH', 'Park']
        train_names = ['Thomas', 'Gordon', 'Emily', 'James', 'Edward',
                       'Percy', 'Henry']
        self.length = 14 # in miles
        self.stop_distance = 2 # stops are 2 miles apart
        self.max_speed = max_speed # miles/min
        self.stops = {} # maps locations to stop names
        self.trains = {} # map trains to location

        self.stopped = {} # maps trains to their stopped state

        self.history = History(train_names)
        self.time = 0
        for i in range(int(self.length/self.stop_distance)):
            stop_loc = i*self.stop_distance
            self.stops[stop_loc] = stop_names[i]
            self.trains[train_names[i]] = stop_loc

            self.stopped[train_names[i]] = 0 # list of trains and their initial stopped state of 0 (false)

            self.history.add_loc(train_names[i], stop_loc)
        self.history.add_stops(self.stops.keys())

    def next_loc(self, loc):
        """
        Args:
            loc (float): a location along the circular track (value between 0 and self.length)
        Returns:
            new_loc (float): the location at the next timestep given the current location `loc`
        """
        new_loc = round(loc + self.max_speed, 2)
        return new_loc % self.length

    def stop_between(self, old_loc, new_loc):
        """
        Args:
            old_loc (float): the previous location
            new_loc (float): the next location
        Returns:
            Returns the stop between the old and new location if there is one, otherwise returns None.
        """
        for stop in self.stops:
            if stop > old_loc and stop <= new_loc:
                return stop
            if new_loc < old_loc:
                if stop >= 0 and stop <= new_loc:
                    return stop
        return None

    def move_train(self, train, time, verbose):
        """
        Moves the train one time step in the simulation.

        Args:
            train (str): the train to move
            time (int): the simulation timestep
            verbose: flag to print outputs
        """
        def ahead_and_too_close(x, y):
            """
            Check for enforcing the trains' minimum distance requirement.
            """
            dist = (y - x) % self.length
            return 0 < dist <= 0.5
        # get location and candidate next location
        old_loc = self.trains[train]
        loc = self.next_loc(old_loc)
        # handle train spacing
        for other_train in self.trains:
            if other_train == train:
                continue
            if ahead_and_too_close(loc, self.trains[other_train]):
                if verbose > 0:
                    print(f'{train} is stuck behind {other_train}')
                loc = old_loc # Don't move train
                self.history.add_loc(train, loc)
                if verbose > 1:
                    print(f'{train} is at location {loc}')
                return

        # checking if train is stopped
        if self.stopped[train] >= 1:
            self.stopped[train] -= 1
            loc = old_loc
            self.history.add_loc(train, loc)
            return

        # handle stops
        if old_loc not in self.stops: # not starting at a stop
            passed_stop = self.stop_between(old_loc, loc)
            if passed_stop != None: # if reaching a stop
                loc = passed_stop # snap to center of stop
                stop = self.stops[loc]

                self.stopped[train] = 2 # if reaches stop, make wait 2 steps

                if verbose > 0:
                    print(f'{train} has arrived at {stop}. Time = {time}')

        self.trains[train] = loc # move train to new loc
        self.history.add_loc(train, loc)
        if verbose > 1:
            print(f'{train} is at location {loc}')

    def move_trains(self, time, verbose):
        """
        Moves all of the trains in the track for one time step.

        Args:
            time (int): the simulation timestep
            verbose: flag to print outputs
        """
        for train in self.trains:
            self.move_train(train, time, verbose)
        self.time += 1
        if verbose > 1:
            print('')

    def get_trains(self):
        """
        Returns a list of all the train names.
        """
        trains = []
        for k in self.trains:
            trains.append(k)
        return trains

    def get_stops(self):
        """
        Returns a list of tuples of (stop name, stop location).
        """
        stops = []
        for k in self.stops:
            stops.append((self.stops[k], k))
        return stops

    def get_history(self):
        """
        Returns the track's history.
        """
        return self.history

    def get_name(self):
        """
        Returns details about the type and speed of track.
        """
        return f'Consistent Speed Track\nSpeed = {60*self.max_speed} MPH'

    def __str__(self):
        output = ''
        for t in self.trains:
            output += f'Train {t} is at location {self.trains[t]}\n'
        return output

class GaussianSlowdownTrack(PerfectTrack):
    """
    A GaussianSlowdownTrack is a discrete-time simulation of a one track circular train line
    in which trains are slowed down at each step by the absolute value of a Gaussian with a
    mean of 0 and a standard deviation of `sigma`.
    """
    def __init__(self, max_speed, sigma):
        super().__init__(max_speed)
        self.sigma = sigma

    def next_speed(self):
        speed_decrease = random.gauss(0, self.sigma)
        temp_speed = self.max_speed - abs(speed_decrease)
        return max(0, temp_speed)

    def next_loc(self, loc):
        """
        Args:
            loc (float): a location along the circular track (value between 0 and self.length)
        Returns:
            new_loc (float): the location at the next timestep given the current location `loc`
        """
        # function to move train with slowed down speed
        new_speed = self.next_speed()
        new_loc = round(loc + new_speed, 2)
        return new_loc % self.length

    def get_name(self):
        """
        Returns details about the type and speed of track.
        """
        return f'Gaussian Slowdown Track\nSpeed = {60*self.max_speed} MPH\nSigma = {60*self.sigma:.2f} MPH'

class SlowZoneTrack(PerfectTrack):
    """
    A SlowZoneTrack is a discrete-time simulation of a one track circular train line
    in which trains move `slow_zone_factor` times slower in the slow zone (between the Charles-MGH
    and Park Street stops).
    """
    def __init__(self, max_speed, slow_zone_factor):
        super().__init__(max_speed)
        self.slow_zone_factor = slow_zone_factor

    def get_name(self):
        """
        Returns details about the type and speed of track.
        """
        return f'Slow Zone Speed Track\nSpeed = {60*self.max_speed} MPH\nSlow Zone Speed = {60 * self.max_speed * self.slow_zone_factor}'

    def in_slow_zone(self, loc):
        # if train loc in between charles-mgh and park, slow down
        if loc >= 10 and loc < 12:
            return True
        return False

    def next_loc(self, loc):
        """
        Args:
            loc (float): a location along the circular track (value between 0 and self.length)
        Returns:
            new_loc (float): the location at the next timestep given the current location `loc`
        """
        #adjusting train speed based on if in slow zone
        if self.in_slow_zone(loc):
            new_speed = self.max_speed * self.slow_zone_factor
            new_loc = round(loc + new_speed, 2)
        else:
            new_loc = round(loc + self.max_speed, 2)
        return new_loc % self.length

def simulate_trains(track_type, num_sims, num_steps, max_speed, slow_down_param, verbose):
    """
    Runs `num_sims` simulations of a track of `track_type` (PerfectTrack, GaussianSlowdownTrack, or
    SlowZoneTrack) that has maximum speed `max_speed` and a `slow_down_param` value, if applicable.
    Each simulation runs for `num_steps` time steps.
    Returns the list of histories from each simulation and the track used.

    Args:
        track_type: the track's object type (i.e., PerfectTrack, GaussianSlowdownTrack, or SlowZoneTrack)
        num_sims (int): the number of simulations
        num_steps (int): the number of time steps per simulation
        max_speed (float): the maximum speed in the track
        slow_down_param (float): None for a PerfectTrack,
                the sigma value for a GaussianSlowdownTrack,
                or the slow_zone_factor for a SlowZoneTrack
        verbose: flag to print outputs

    Returns:
        a tuple (histories, track)
            histories: list of histories
            track: any of the track instances used when generating the histories
    """
    histories = []

    #choosing which type of simulation track
    for _ in range(num_sims):
        if track_type == PerfectTrack:
            sim = PerfectTrack(max_speed)
        elif track_type == GaussianSlowdownTrack:
            sim = GaussianSlowdownTrack(max_speed, slow_down_param)
        else:
            sim = SlowZoneTrack(max_speed, slow_down_param)

        #running simulation for the desired number of steps
        for time in range(num_steps):
            sim.move_trains(time, verbose)
        histories.append(sim.get_history())

    return (histories, sim)


############################################################
# Part 2: Plotting #
############################################################

############## DO NOT MODIFY THESE FUNCTIONS ###############

def label_plot(title, x_label, y_label, legend = False):
    """
    Labels the plot with the specified title and axes labels.
    """
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend:
        plt.legend()

def plot_distributions(track_type, histories, max_speed, slow_down_param, alpha):
    """
    Creates a histogram showing the distribution of interarrival times.

    Args:
        track_type: the track's object type (i.e. PerfectTrack, GaussianSlowdownTrack, or SlowZoneTrack)
        histories (list): list of histories
        max_speed (float): the maximum speed in the track
        slow_down_param (float): None for a PerfectTrack,
                the sigma value for a GaussianSlowdownTrack,
                or the slow_zone_factor for a SlowZoneTrack
        alpha (float): value between 0-1, controls the transparency of the histogram bars
    """
    mean_wait, std, wait_times = analyze_histories(histories)
    vals, bins = np.histogram(wait_times, bins = 10)
    total = sum(vals)
    percentages = (vals / total) * 100
    filtered_percents = percentages[percentages > 0.25]
    filtered_bins = bins[1:][percentages > 0.25]
    if track_type == GaussianSlowdownTrack:
        plt.bar(filtered_bins, filtered_percents, alpha = alpha,
                label = f'With max speed = {60*max_speed:.2f} MPH\n' +
                        f'std slowdown = {60*slow_down_param:.2f} MPH\n' +
                        f'mean inter-arrival time = {mean_wait:.2f} min.')
    elif track_type == SlowZoneTrack:
        plt.bar(filtered_bins, filtered_percents, alpha = alpha,
                label = f'With speed = {60*max_speed:.2f} MPH\n' +
                        f'Slow zone factor = {slow_down_param}\n' +
                        f'mean inter-arrival time = {mean_wait:.2f} min.')
    elif track_type == PerfectTrack:
        plt.bar(filtered_bins, filtered_percents, alpha = alpha,
                label = f'With fixed speed = {60*max_speed:.2f} MPH\n' +
                        f'mean inter-arrival time = {mean_wait:.2f} min.')
    label_plot('Observed Time Between Trains', 'Minutes',
              'Percentage of Arrivals', True)
############################################################
##################### YOUR CODE BELOW ######################

PLOT_MAX_SPEED_MPH = 30
PLOT_MAX_SPEED = PLOT_MAX_SPEED_MPH / 60
NUM_SIMS = 32
NUM_STEPS = 1000
SEED = 6100
SIGMA_MPH_VALUES = np.arange(0, 25, 1)
SLOW_ZONE_VALUES = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25,
                             0.2, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02])

def analyze_histories(histories):
    """
    Given a list of histories, computes the mean and standard deviation of all the interarrival times
    from those histories.

    Args:
        histories (list): list of histories
    Returns:
        a tuple (mean, std, all_interarrival_times)
            mean: the average interarrival time across all the histories in histories
            std: the standard deviation of interarrival times across all the histories in histories
            all_interarrival_times: a list of all interarrival times from all the histories in histories
    """
    all_interarrival_times = []

    # iterate through histories and stop locations
    for history in histories:
        for stop in history.get_stop_locs():
            all_arrivals = []

            # for each train check if at stop and add arrival time to list
            for train in history.get_trains():
                at_stop = False
                for time, loc in enumerate(history.get_train_locs(train)):
                    is_at_stop = abs(loc - stop) < .01
                    if is_at_stop and not at_stop and time != 0:
                        all_arrivals.append(time)
                    at_stop = is_at_stop

            all_arrivals.sort()

            # calculate all interarrival times
            interarrvial_times = []
            for i in range(len(all_arrivals) - 1):
                interarrvial_times.append(all_arrivals[i+1] - all_arrivals[i])

            all_interarrival_times.extend(interarrvial_times)

    # calculate mean and standard deviation based on interarrival times
    mean = np.mean(all_interarrival_times)
    std = np.std(all_interarrival_times)

    return (mean, std, all_interarrival_times)

def sweep_metric_vs_sigma(max_speed_mph, num_sims, num_steps, sigma_mph_values):
    """
    Sweeps sigma values (in MPH) for GaussianSlowdownTrack and returns means/stds.
    """
    means = []
    stds = []
    max_speed = max_speed_mph / 60

    for sigma_mph in sigma_mph_values:
        sigma = sigma_mph / 60
        histories, _ = simulate_trains(
            GaussianSlowdownTrack, num_sims, num_steps, max_speed, sigma, 0
        )
        mean, std, _ = analyze_histories(histories)
        means.append(mean)
        stds.append(std)

    return sigma_mph_values, means, stds

def sweep_metric_vs_slow_zone_factor(max_speed_mph, num_sims, num_steps, slow_zone_values):
    """
    Sweeps slow_zone_factor values for SlowZoneTrack and returns means/stds.
    """
    means = []
    stds = []
    max_speed = max_speed_mph / 60

    for slow_zone_factor in slow_zone_values:
        histories, _ = simulate_trains(
            SlowZoneTrack, num_sims, num_steps, max_speed, slow_zone_factor, 0
        )
        mean, std, _ = analyze_histories(histories)
        means.append(mean)
        stds.append(std)

    return slow_zone_values, means, stds

def make_mean_plot(x_values, y_values, use_sigma):
    """
    Plots mean inter-arrival time against sigma or slow zone factor.
    """
    plt.figure()
    plt.plot(x_values, y_values)
    if use_sigma:
        label_plot(
            f'Mean Inter-arrival Time vs. Sigma\n'
            f'Max Speed = {PLOT_MAX_SPEED_MPH:.2f} MPH, {NUM_SIMS} trials',
            'Sigma (MPH)',
            'Minutes'
        )
    else:
        label_plot(
            f'Mean Inter-arrival Time vs. Slow Zone Factor\n'
            f'Max Speed = {PLOT_MAX_SPEED_MPH:.2f} MPH, {NUM_SIMS} trials',
            'Slow Zone Factor',
            'Minutes'
        )

def make_std_plot(x_values, y_values, use_sigma):
    """
    Plots std deviation of inter-arrival time against sigma or slow zone factor.
    """
    plt.figure()
    plt.plot(x_values, y_values)
    if use_sigma:
        label_plot(
            f'Standard Deviation of Arrivals vs. Sigma\n'
            f'Max Speed = {PLOT_MAX_SPEED_MPH:.2f} MPH',
            'Sigma (MPH)',
            'Minutes'
        )
    else:
        label_plot(
            f'Standard Deviation of Arrivals vs. Slow Zone Factor\n'
            f'Max Speed = {PLOT_MAX_SPEED_MPH:.2f} MPH',
            'Slow Zone Factor',
            'Minutes'
        )

def generate_required_plots():
    """
    Generates the 6 required plots in the specified screenshot order.
    """
    sigma_x, sigma_means, sigma_stds = sweep_metric_vs_sigma(
        PLOT_MAX_SPEED_MPH, NUM_SIMS, NUM_STEPS, SIGMA_MPH_VALUES
    )
    slow_zone_x, slow_zone_means, slow_zone_stds = sweep_metric_vs_slow_zone_factor(
        PLOT_MAX_SPEED_MPH, NUM_SIMS, NUM_STEPS, SLOW_ZONE_VALUES
    )

    # Mean vs Sigma
    make_mean_plot(sigma_x, sigma_means, True)

    # Mean vs Slow Zone Factor
    make_mean_plot(slow_zone_x, slow_zone_means, False)

    # Std vs Sigma
    make_std_plot(sigma_x, sigma_stds, True)

    # Std vs Slow Zone Factor
    make_std_plot(slow_zone_x, slow_zone_stds, False)

    # Distribution for sigma = 12 MPH
    plt.figure()
    sigma_histories, _ = simulate_trains(
        GaussianSlowdownTrack, NUM_SIMS, NUM_STEPS, PLOT_MAX_SPEED, 12 / 60, 0
    )
    plot_distributions(
        GaussianSlowdownTrack, sigma_histories, PLOT_MAX_SPEED, 12 / 60, 0.8
    )

    # Distribution for slow_zone_factor = 0.5
    plt.figure()
    slow_zone_histories, _ = simulate_trains(
        SlowZoneTrack, NUM_SIMS, NUM_STEPS, PLOT_MAX_SPEED, 0.5, 0
    )
    plot_distributions(
        SlowZoneTrack, slow_zone_histories, PLOT_MAX_SPEED, 0.5, 0.8
    )

def plot_reducing_variability_comparison():
    """
    Creates one comparison histogram for:
    - max speed 30 MPH with sigma 12 MPH
    - max speed 24 MPH with sigma 4 MPH
    """
    plt.figure()

    max_speed_1 = 30 / 60
    sigma_1 = 12 / 60
    histories_1, _ = simulate_trains(
        GaussianSlowdownTrack, NUM_SIMS, NUM_STEPS, max_speed_1, sigma_1, 0
    )
    plot_distributions(
        GaussianSlowdownTrack, histories_1, max_speed_1, sigma_1, 0.7
    )

    max_speed_2 = 24 / 60
    sigma_2 = 4 / 60
    histories_2, _ = simulate_trains(
        GaussianSlowdownTrack, NUM_SIMS, NUM_STEPS, max_speed_2, sigma_2, 0
    )
    plot_distributions(
        GaussianSlowdownTrack, histories_2, max_speed_2, sigma_2, 0.7
    )

    plt.title(f'Observed Time Between Trains, {NUM_SIMS} trials')

def show_all_plots():
    """
    Runs all requested plots and displays all figures.
    """
    generate_required_plots()
    plot_reducing_variability_comparison()
    plt.show()

# PLOTTING CODE: You may define any helper functions above and
# place any plotting or testing code in the if __name__ == "__main___" block below.

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    show_all_plots()

    ## Feel free to uncomment and modify the below example for PerfectTrack to visualize your code
    # num_sims = 32
    # num_steps = 500
    # max_speed = 0.5

    # p_histories, p_track = simulate_trains(PerfectTrack, num_sims, num_steps, max_speed, None, 0)

    # frame_rate = 4 # frames/second
    # ani = produce_animation(p_histories[0], p_track)
    # ani.save('constant-speed.gif', writer = 'ffmpeg', fps = frame_rate)
