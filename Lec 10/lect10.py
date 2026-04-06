#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:19:30 2019

@author: johnguttag
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import sys

############################################################
### DISPLAY DEFAULTS FOR PLOTTING AND DATAFRAMES
############################################################

# Set up some global plot parameters to make plots look better
# set line width
plt.rcParams['lines.linewidth'] = 3
# set font size for titles
plt.rcParams['axes.titlesize'] = 20
# set font size for labels on axes
plt.rcParams['axes.labelsize'] = 20
# set size of numbers on x-axis
plt.rcParams['xtick.labelsize'] = 15
# set size of numbers on y-axis
plt.rcParams['ytick.labelsize'] = 15
# set size of ticks on x-axis
plt.rcParams['xtick.major.size'] = 7
# set size of ticks on y-axis
plt.rcParams['ytick.major.size'] = 7
# set size of markers
plt.rcParams['lines.markersize'] = 8
# set values for legend
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.handlelength'] = 0.5


############################################################
# PROBABILITY DISTRIBUTIONS
############################################################


########################################
# Discrete Uniform distribution

# num_samples = 1000
# random.seed(0)
# a, b = 2, 11
# for trial in range(2):
#     uniform_vals = [random.randint(a, b) for _ in range(num_samples)]
#     counts, bins = np.histogram(uniform_vals, bins = b-a+1)
#     prob_dist = counts/num_samples
#     x_vals = [(bins[i] + bins[i-1])/2 for i in range(1, len(bins))]
#     plt.bar(x_vals, prob_dist, label = f'trial {trial}', alpha = 0.5)
# plt.xlabel('Outcome value')
# plt.xlim(1, 12)
# plt.xticks(range(1, 13))
# plt.ylabel('Probability density')
# plt.title(f'Discrete uniform distribution on [{a}, {b}]')
# plt.legend()


########################################
# Normal/Gaussian distribution

# mu, sigma = 20, 3
# num_samples = 100_000
# num_bins = 25
# normal_vals = [random.gauss(20, 3) for _ in range(num_samples)]
# counts, bins = np.histogram(normal_vals, bins = num_bins)
# prob_dist = counts/num_samples
# x_vals = [(bins[i] + bins[i-1])/2 for i in range(1, len(bins))]
# plt.bar(x_vals, prob_dist)
# plt.xlabel('Outcome value')
# plt.ylabel('Probability density')
# plt.title(f'Normal distribution with $\mu={mu}$ and $\sigma = {sigma}$')


########################################
# # Exponential distribution

# lam = 0.5
# num_samples = 100_000
# num_bins = 25
# exp_vals = [random.expovariate(lam) for _ in range(num_samples)]
# counts, bins = np.histogram(exp_vals, bins = num_bins)
# prob_dist = counts/num_samples
# x_vals = [(bins[i] + bins[i-1])/2 for i in range(1, len(bins))]
# plt.bar(x_vals, prob_dist)
# plt.xlim(0, 15)
# plt.xlabel('Outcome value')
# plt.ylabel('Probability density')
# plt.title(f'Exponential distribution with $\lambda={lam}$')

# sys.exit()


########################################
# Bernoulli distribution

# num_samples = 1_000
# num_bins = 2
# prob = 0.3
# bern_vals = [random.choices([0, 1], weights=[1-prob, prob], k=num_samples)]
# counts, bins = np.histogram(bern_vals, bins = num_bins)
# prob_dist = counts/num_samples
# plt.bar((0, 1), prob_dist)
# plt.xticks((0, 1))
# plt.xlabel('Outcome value')
# plt.ylabel('Probability density')
# plt.title(f'Bernoulli distribution with prob = {prob}')

# sys.exit()

# ########################################
# # Binomial distribution

# num_samples = 1_000
# num_trials = 1_000
# num_bins = 11
# prob = 0.3
# binom = np.array(random.choices([0, 1], weights=[1-prob, prob], k=num_samples))
# for i in range(num_trials - 1):
#     bern = np.array(random.choices([0, 1], weights=[1-prob, prob], k=num_samples))
#     binom += bern

# print(f'Mean = {sum(binom)/len(binom)}, Std = {np.std(binom)}')

# counts, bins = np.histogram(binom, bins = num_bins)
# bin_centers = np.round((bins[:-1] + bins[1:])/2)
# prob_dist = counts/num_samples
# plt.bar(bin_centers, prob_dist, width = 5.0)
# plt.xticks(bin_centers, rotation = 45)
# plt.xlabel('Outcome value')
# plt.ylabel('Probability density')
# plt.title(f'Binomial distribution with prob = {prob}\n'
#           f'Sample size = {num_samples:,}, Trials = {num_trials:,}')

# sys.exit()



# A demo showing that many random binary choices leads to a normal distribution

# %matplotlib qt
# from matplotlib.animation import FuncAnimation

# def galton_board(num_slots, num_balls):
#     slots = [0] * (num_slots)  # Create slots to collect balls

#     for _ in range(num_balls):
#         pos = 0  # Start at the top of the board
#         for _ in range(num_slots): # number of levels
#             if random.random() < 0.5:  # 50% chance to move right
#                 pos += 1
#         slots[pos] += 1  # Increment the corresponding slot

#         yield slots

# def update(frame):
#     plt.cla()
#     plt.bar(range(len(frame)), frame)
#     plt.xlabel('Slots', fontsize = 36)
#     plt.ylabel('Number of Balls', fontsize = 36)
#     plt.title(f'Galton Board Simulation with {num_balls:,} balls', fontsize = 36)

# random.seed(1)

# num_slots = 10  # One more than number of slots in the Galton board
# num_balls = 10  # Number of balls to drop
# num_balls = 1000  # Number of balls to drop
# if num_balls < 50:
#     interval = 500
# else:
#     interval = 0.00001
# results = galton_board(num_slots, num_balls)

# fig = plt.figure()
# ani = FuncAnimation(fig, update, frames=results, interval=interval,
#                     cache_frame_data=False, repeat=False)
# # Toggle fullscreen mode
# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()

# plt.show()

# sys.exit()

############################################################
# CENTRAL LIMIT THEOREM ON DICE ROLLS
############################################################


def roll_dice(n, sides, trials):
    """Run trials of summing n dice with given number of sides
       Return an array of sums of length trials"""
    sums = []
    for i in range(trials):
        tot = 0
        for j in range(n):
            tot = tot + random.randint(1, sides)
        sums.append(tot)
    return sums



#######################################
# Run experiments summing up increasing numbers of 6-sided die rolls
# Compare the mean, stdev values to those predicted by CLT

# random.seed(0)

# dice = (1, 5, 10, 100, 1_000_000)
# num_trials = 10_000
# num_trials = 100

# print(f'Empirical values of mean and stdev, {num_trials:,} trials')
# for num_dice in dice:
#     dice_means = np.array(roll_dice(num_dice, 6, num_trials)) / num_dice
#     mean = np.mean(dice_means)
#     stdev = np.std(dice_means)
#     print(f'mean of {num_dice:,} rolls, mu={mean:.3f}, sigma={stdev:.3f}')

# print()
# print("Theoretical values of mean and stdev, according to CLT")
# for num_dice in dice:
#     die_mean = sum(range(1,7))/6
#     die_variance = (6**2 - 1) / 12 # discrete uniform dist with 6 values
#     sum_mean = die_mean * num_dice
#     sum_variance = die_variance * num_dice
#     mean_mean = sum_mean / num_dice
#     mean_variance = sum_variance / num_dice**2
#     mean_stdev = mean_variance**0.5
#     print(f'mean of {num_dice:,} rolls, mu={mean_mean:.3f}, '
#           f'sigma={mean_stdev:.3f}')
    
# sys.exit()

#### Generate plots showing convergence rates

def plot_convergence(dist_fcn, mu, dist_label, num_trials, max_n):
    sample_sizes = np.arange(1, max_n + 1)
    errors = np.zeros(max_n)
    for _ in range(num_trials):
        vals = dist_fcn(max_n)
        mean_val = np.cumsum(vals)/sample_sizes
        errors += np.abs((mean_val - mu / mu))
    errors /= num_trials
    plt.plot(sample_sizes, errors, label = dist_label)


max_n = 2000
num_trials = 1000

# uniform = lambda max_n: np.random.uniform(0, 2, max_n)
# uniform_label = 'Uniform(0, 2)'
# plot_convergence(uniform, 1, uniform_label, num_trials, max_n)

# norm_1 = lambda max_n: np.random.normal(1, 1, max_n)
# norm_1_label = 'Normal(1, 1)'
# plot_convergence(norm_1, 1, norm_1_label, num_trials, max_n)

# exp = lambda max_n: np.random.exponential(1, max_n)
# exp_label = 'Exponentional(mean = 1)'
# plot_convergence(exp, 1, exp_label, num_trials, max_n)

# for sigma in (1.0, 0.75, 0.5, 0.25):
#     norm = lambda max_n: np.random.normal(1, sigma, max_n)
#     label = f'Normal(1, {sigma})'
#     plot_convergence(norm, 1, label, num_trials, max_n)
    
# plt.semilogy()
# ax = plt.gca()
# ax.set_yticks((1, 0.1, 0.01))
# plt.xlabel('Sample Size')
# plt.ylabel('Mean Absolute Difference')
# plt.title(f'Sample Mean vs Population Mean\n Mean of {num_trials:,} Trials')
# plt.legend()

# exp_vals = np.random.exponential(1, max_n)
# counts, bins = np.histogram(exp_vals, bins = 25)
# prob_dist = counts/max_n
# x_vals = [(bins[i] + bins[i-1])/2 for i in range(1, len(bins))]
# plt.bar(x_vals, prob_dist)
# plt.xlim(0, 6)
# plt.xlabel('Outcome value')
# plt.ylabel('Probability density')
# plt.title('Exponential distribution with mean = 1')

# sys.exit()


# ############################################################
# ### Pandas
# ############################################################


# ############################################################
# ### EXAMPLE 1: WOMEN'S WORLD CUP
# ############################################################

# ########################################
# ### CSV import, dataframe attributes
# ########################################

# wwc = pd.read_csv('wwc2022_qf.csv')
# print(wwc)
# print(wwc.to_string())

# print(wwc.columns)

# print(wwc.index)

# print(f'type of columns: {type(wwc.columns)}')
# print(f'type of index: {type(wwc.index)}')
# for c in wwc.columns:
#     print(c)
# for i in wwc.index:
#     print(i)

# print(wwc.values)
# print(type(wwc.values))

# sys.exit()


# ########################################
# ### Constructing dataframes
# ########################################

# print(pd.DataFrame())
# rounds = ['Semis', 'Semis', '3rd Place', 'Championship']
# teams = ['Spain', 'England', 'Sweden', 'Spain']
# df = pd.DataFrame({'Round': rounds, 'Winner': teams})
# print(df)

# df['W Goals'] = [2, 3, 2, 1]
# print(df)

# df_new = df.drop('Winner', axis = 'columns', inplace = False)
# print(df_new, '\n')
# print(df, '\n')
# df.drop('Winner', axis = 'columns', inplace = True)
# print(df, '\n')

# # Add column
# df['Winner'] = teams
# print(df)

# # Add multiple rows
# quarters_dict = {'Round': ['Quarters'] * 4,
#                   'Winner': ['Spain', 'Sweden', 'England', 'Australia'],
#                   'W Goals': [2, 2, 2, 0]}
# # df = pd.concat([pd.DataFrame(quarters_dict), df], sort = False)
# df = pd.concat([pd.DataFrame(quarters_dict), df], sort = False,
#                ignore_index = True)
# print(df.to_string())

# # sys.exit()
# # df = df.reset_index(drop = True)
# # print(df.to_string())

# # df = df.set_index('Round')
# # print(df.to_string())

# sys.exit()

# # ########################################
# # ### Simple selection mechanisms
# # ########################################

# wwc = pd.read_csv('wwc2022_qf.csv')

# winners = ''
# for w in wwc['Winner']:
#     winners += w + ', '
# print(winners[:-2])

# row_2 = ''
# for r in (wwc.iloc[2]):
#     row_2 += str(r) + ', '
# print(row_2[:-2])

# sys.exit()
# ########################################
# ### Plain (mostly column) indexing
# ########################################

# print(wwc.to_string())

# print(wwc['Winner'])
# print(type(wwc['Winner']))
# for w in wwc['Winner']:
#     print(w)

# print(wwc[['Winner', 'Loser']])
# print(wwc[['Round','Winner','Loser','W Goals','L Goals']])

# print(wwc[1])                   # KeyError, no column labeled 1
# print(wwc[1:2])                 # slicing selects rows
# wwc[1] = [1,2,3,4,5,6,7,8]      # add explicit column labeled 1
# print(wwc[1])

# print(wwc['L Goals'][7])
# wwc['L Goals'][7] = 1
# print(wwc.to_string())

########################################
### Label indexing
########################################

# print(wwc.to_string())

# print(wwc.loc[2])
# print(type(wwc.loc[2]))

# print(wwc.loc[[1,3,5]])
# print(wwc.loc[3:7:2])
# print(wwc.loc[6:])
# print(wwc.loc[:2])
# print(list(range(5))[:2])
# print(wwc.loc[5:5])
# print(type(wwc.loc[5:5]))

# print(wwc.loc[2, 'Winner'])
# print(wwc.loc[0:2, ['Winner', 'Loser']])
# print(wwc.loc[0:2, 'Round':'L Goals':2])

# wwc['L Goals'][7] = 1
# wwc.loc[7, 'L Goals'] = 1
# print(wwc.to_string())

# wwc_by_round = wwc.set_index('Round')
# print(wwc_by_round.to_string())

# print(wwc_by_round.loc['Semis'])
# print(wwc_by_round.loc[['Semis', 'Championship']])
# print(wwc_by_round.loc['Quarters':'Semis':2])
# wwc_by_round.loc['Semis', 'Winner'] = 'Canada'
# print(wwc_by_round.to_string())

########################################
### Grouping
########################################

# grouped_by_round = wwc.groupby('Round', sort = False)
# print(grouped_by_round.sum())

# print(wwc.groupby('Winner').mean())
# print(wwc.groupby(['Loser', 'Round']).mean().to_string())

########################################
### Boolean selection
########################################

# print(wwc.loc[wwc['Winner'] == 'Sweden'])
# print((wwc['Winner'] == 'Sweden').to_string())
# print(((wwc['Winner'] == 'Sweden') | (wwc['Loser'] == 'Sweden')).to_string())
# print(wwc.loc[(wwc['Winner'] == 'Sweden') | (wwc['Loser'] == 'Sweden')])

def get_country(df, country):
    """df a DataFrame with series labeled Winner and Loser
       country a str
       returns a DataFrame with all rows in which country appears
       in either the Winner or Loser column"""
    return df.loc[(df['Winner'] == country) | (df['Loser'] == country)]

# print(get_country(get_country(wwc, 'Sweden'),'Germany'))

def get_games(df, countries):
    return df[(df['Winner'].isin(countries)) |
              (df['Loser'].isin(countries))]

# print(2 * np.array([1,2,3]))
# print(2 * wwc['W Goals'])

# print(wwc['W Goals'].sum())
# print(wwc[wwc['Winner'] == 'Sweden']['W Goals'].sum() +
#       wwc[wwc['Winner'] == 'Sweden']['L Goals'].sum())
# print((wwc['W Goals'].sum() - wwc['L Goals'].sum()) / len(wwc.index))

# wwc['G Diff'] = wwc['W Goals'] - wwc['L Goals']
# new_row_dict = {'Round': ['Total'],
#                 'W Goals': [wwc['W Goals'].sum()],
#                 'L Goals': [wwc['L Goals'].sum()],
#                 'G Diff': [wwc['G Diff'].sum()]}
# new_row = pd.DataFrame(new_row_dict)
# wwc = pd.concat([wwc, new_row]).reset_index(drop = True)
# print(wwc.to_string())
# print(wwc.loc[wwc['Round'] != 'Total'].corr(method = 'pearson'))

############################################################
### EXAMPLE 2: RISING temps
############################################################

# # print(pd.get_option('display.max_rows'))
# # print(pd.get_option('display.max_columns'))
# pd.set_option('display.max_rows', 6)
# pd.set_option('display.max_columns', 5)

# ############################################################
# ### BASIC DATA TYPES
# ############################################################

# # temps = pd.read_csv('US_temps.csv')
# # print(temps)
# # print(type(temps))
# # print(temps['Boston'])
# # print(type(temps['Boston']))

# # plt.figure(figsize = (14, 5)) # set aspect ratio for figure
# # plt.plot(temps['Boston'])
# # plt.title('Historical Daily Temperature for Boston')
# # plt.xlabel('Days Since 1/1/1961')
# # plt.ylabel('Degrees C')

########################################
### Calculate and plot mean temps
########################################

# temps = pd.read_csv('US_temperatures.csv')
# # print(temps)

# # Make date the index, reformat dates, change temp to F
# temps['Date'] = pd.to_datetime(temps['Date'], format='%Y%m%d')
# temps.set_index('Date', inplace = True)
# temps[temps.columns] = temps[temps.columns]*9/5 + 32
# print(temps)

# sys.exit()

# # # Look at a specific date
# # print(temps.loc['2015-12-31'].to_string())

# # Look at a specific date/cities
# # print(temps.loc['2015-12-31', ['Boston', 'Tampa']])

# # Add some statistics for each date
# temps['Max T'] = temps.max(axis = 'columns')
# temps['Min T'] = temps.min(axis = 'columns')
# temps['Mean T'] = temps.mean(axis = 'columns').round(2)
# # print(temps)

# plt.figure(figsize = (14, 5))
# plt.plot(list(temps['Mean T']))
# # plt.plot(list(temps['Mean T'])[0:3*365])
# plt.title('Mean Temp Across 21 US Cities')
# plt.xlabel('Days Since 1/1/1961')
# plt.ylabel('Degrees F')

# sys.exit()
########################################
### Calculate mean temps for each year
########################################

temps = pd.read_csv('US_temperatures.csv')
temps.set_index('Date', inplace = True)
temps['Max T'] = temps.max(axis = 'columns')
temps['Min T'] = temps.min(axis = 'columns')
temps['Mean T'] = temps.mean(axis = 'columns').round(2)

temps.reset_index(drop = False, inplace = True)
temps['Year'] = temps['Date'].apply(lambda d: str(d)[0:4])
temps = temps.loc[:, ['Year', 'Min T', 'Max T', 'Mean T']]
print(temps)
grouped_temps = temps.groupby('Year').mean()
print(grouped_temps)

years = [str(yr) for yr in range(1961, 2016)]
yearly_means = [round(grouped_temps.loc[yr]['Mean T'].mean(), 2)
                for yr in years]
plt.figure(figsize = (10, 5))
plt.plot(years, yearly_means)
plt.title('Mean Annual Tmep. in 21 U.S. Cities')
plt.ylabel('Degrees F')
plt.xticks(range(0, len(years), 4), rotation = 'vertical')

sys.exit()


########################################
### Plot extremal temperature ranges per city
########################################

def r_squared(observed, predicted):
    error = ((predicted - observed)**2).sum()
    mean_error = error / len(observed)
    return 1 - mean_error / np.var(observed)

temps = pd.read_csv('US_temperatures.csv')
temps.drop('Date', axis = 'columns', inplace = True)
temps[temps.columns] = temps[temps.columns]*9/5 + 32
min_max = pd.DataFrame({'Min': temps.min(), 'Max': temps.max()})
print(min_max)

min_max['Range'] = min_max['Max'] - min_max['Min']
min_max_sorted = min_max.sort_values('Range', ascending=False)
plt.figure(figsize=(12,6))
plt.plot(min_max_sorted.index, min_max_sorted['Max'], 'ro',
         label='Max Temp')
plt.plot(min_max_sorted.index, min_max_sorted['Min'], 'bo',
         label='Min Temp')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Temperature (°F)')
plt.title('Min and Max Mean Daily Temperatures for Each City' +
          '\n1961- 2015')
plt.legend(loc = 'lower right')
plt.tight_layout()

model_1 = np.polyfit(min_max_sorted['Min'], min_max_sorted['Max'], 1)
print(f'model = {round(model_1[0], 3)}*x + {round(model_1[1], 3)}')
preds = np.polyval(model_1, min_max_sorted['Min'])
plt.plot(preds, label = 'Linear fit')
plt.legend()
r2 = round(r_squared(min_max_sorted['Max'], preds), 6)
print(f'r-squared = {r2}')
cor = round(min_max_sorted['Min'].corr(min_max_sorted['Max']),6)
print(f'correlation = {cor}')


########################################
### Look at fossil fuel consumption
########################################

# emiss = pd.read_csv('global-fossil-fuel-consumption.csv')
# emiss['Total'] = emiss.sum(axis = 'columns')
# emiss['Rolling tot.'] = emiss['Total'].rolling(5).mean()
# print(emiss)
# plt.plot(emiss['Year'], emiss['Total'], label = 'emissions')
# plt.plot(emiss['Year'], emiss['Rolling tot.'], label = '5 yr rolling')
# plt.legend()