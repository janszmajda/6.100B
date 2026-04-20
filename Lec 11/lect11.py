
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.stats as stats
import sys

#set line width
plt.rcParams['lines.linewidth'] = 4
#set font size for titles 
plt.rcParams['axes.titlesize'] = 20
#set font size for labels on axes
plt.rcParams['axes.labelsize'] = 20
#set size of numbers on x-axis
plt.rcParams['xtick.labelsize'] = 16
#set size of numbers on y-axis
plt.rcParams['ytick.labelsize'] = 16
#set size of ticks on x-axis
plt.rcParams['xtick.major.size'] = 7
#set size of ticks on y-axis
plt.rcParams['ytick.major.size'] = 7
#set size of markers
plt.rcParams['lines.markersize'] = 10
# set values for legend
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.handlelength'] = 0.5

### p-values

def gen_incomes(mu, sigma, number):
    return [int(random.gauss(mu, sigma)) for _ in range(number)]

def compare_distributions(d1, d2, verbose = True):
    mean_1, mean_2 = sum(d1)/len(d1), sum(d2)/len(d2)
    t_val, p_val = stats.ttest_ind(d1, d2)
    if verbose:
        label_1 = f'Course 1\n$\mu$ = \${sum(d1)/len(d1):.2f}'
        label_2 = f'Course 6\n$\mu$ = \${sum(d2)/len(d2):.2f}'
        plt.hist(d1, bins = 50, density = True, label = label_1)
        plt.hist(d2, bins = 50, density = True, label = label_2,
                 alpha = 0.8)
        plt.legend(loc = 'best')
        plt.ylabel('Probability Density')
        plt.xlabel('Wage')
        plt.title(f'Hourly Wage (n = {(len(d1) + len(d2)):,})')
        print(f'Difference in means = {abs(mean_1 - mean_2):.4f}, '
              f'test-stat = {t_val:.3f}, p-value = {p_val:.3f}')
    return p_val

def one_tail(d1, d2): # u1 > u2
    t_val, p_val = stats.ttest_ind(d1, d2)
    if t_val < 0:
        p_one_tailed = p_val/2
    else:
        p_one_tailed = 1 - (p_val/2)
    return t_val, p_one_tailed

random.seed(0)
n = 100
mu = 100
sigma = 50

# n = 5_000_000
# mu = 100
# sigma = 10

# d1 = gen_incomes(mu, sigma, n)
# d2 = gen_incomes(mu, sigma, n)
# # d2 = gen_incomes(mu + 0.01, sigma, n)
# compare_distributions(d1, d2)

# # Look at one-tailed tests
# print('Try mean(d1) > mean(d2)')
# print(f'  test-stat = {one_tail(d1, d2)[0]:.3f}, '
#       f'p_val = {one_tail(d1, d2)[1]:.3f}')
# print('Try mean(d2) > mean(d1)')
# print(f'  test-stat = {one_tail(d2, d1)[0]:.3f}, '
#       f'p_val = {one_tail(d2, d1)[1]:.3f}')

# sys.exit()


# # Visualize relation of t-statistic to Empirical Rule

# two_tail = False

# alphas = (0.3173, 0.05, 0.002699) # 1, 1.96, and 2 stds
# for alpha in alphas:
#     crit = stats.norm.ppf(1 - alpha/2)
    
#     x = np.linspace(-4, 4, 1000)
#     y = stats.norm.pdf(x)
    
#     fig, ax = plt.subplots(figsize=(8, 5))
    
#     ax.plot(x, y, color='black', linewidth=2)
#     if two_tail:
#         ax.fill_between(x, y, where=(x <= -crit), color='lightcoral')
#     else:
#         ax.fill_between(x, y, where=(x <= -crit), color='lightgrey', alpha=0.6)
#     ax.fill_between(x, y, where=(x >= crit), color='lightcoral')
    
#     ax.fill_between(x, y, where=((x > -crit) & (x < crit)),
#                     color='lightgray', alpha=0.6)
    
#     if two_tail:
#         ax.axvline(-crit, color='gray', linestyle='--')
#     ax.axvline(crit, color='gray', linestyle='--')
    
#     if two_tail:
#         ax.text(-3.6, 0.09, 'Rejection\nRegion', size = 14, ha='center')
#     ax.text(0, 0.10, 'Acceptance\nRegion', size = 14, ha='center')
#     ax.text(3.6, 0.09, 'Rejection\nRegion', size = 14, ha='center')
#     ax.text(-2.8, 0.05, r'$\alpha/2$', size=16)
#     ax.text(2.4, 0.05, r'$\alpha/2$', size=16)
    
#     if two_tail:
#         ax.set_title(f'Two-Tailed Hypothesis Test\n($\\alpha = {alpha}$)')
#     else:
#         ax.set_title(f'One-Tailed Hypothesis Test\n($\\alpha = {alpha}$)')
#     ax.set_xlabel('t')
#     ax.set_ylabel('Density')
    
#     # Clean up axes
#     ax.set_xlim(-4, 4)
#     ax.set_ylim(0, 0.45)
#     ax.set_xticks([-crit, 0, crit])
#     ax.set_xticklabels([f'{-crit:.2f}', '0', f'{crit:.2f}'])
#     ax.set_yticks([])
    
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
    
#     plt.tight_layout()
#     plt.show()
#     if alpha != alphas[-1]:
#         plt.figure()

# sys.exit()


## Multiple hypotheses

random.seed(3)
# random.seed(0)
# random.seed(1)
sample_size = 300
mu = 100
sigma = 90
dists = {}
num_courses = 8
for course in range(1, num_courses + 1):
    dists[course] = gen_incomes(mu, sigma, sample_size)
p_vals = {}
for c1 in dists:
    for c2 in dists:
        if c1 != c2 and (c2, c1) not in p_vals:
            p_vals[(c1, c2)] = (compare_distributions(dists[c1], dists[c2], False))

any_sig = False
for k in p_vals:
    if p_vals[k] <= 0.05:
        any_sig = True
        mean_diff = (sum(dists[k[0]])/len(dists[k[0]]) -
                    sum(dists[k[1]])/len(dists[k[1]]))
        print(f'The difference between Courses {k[0]} and {k[1]} is'
              f' ${mean_diff:.3f}, with p = {p_vals[k]:.3f}')
if not any_sig:
    print('No significant differences')

x_vals = [f'{k[0]}/{k[1]}' for k in p_vals]
y_vals = [p_vals[k] for k in p_vals]
plt.bar(x_vals, y_vals,
        label = f'Mean p-value = {sum(y_vals)/len(y_vals):.3f}')
plt.xticks(size = 10, rotation = 60)
plt.yticks(size = 12)
plt.title('Distribution of P-values', size = 14)
plt.xlabel('Course Pair', size = 14)
plt.ylabel('P-value', size = 14)
plt.legend()

print(f'Min p-value = {min(y_vals):.3f}')
plt.show()
sys.exit()

### Statistical fallacies

# hr = [1, 1.3, 1.6, 1.9]
# usage = ['None', 'Low', 'Moderate', 'High']
# plt.plot(usage, hr, 'bo')
# plt.title('Tylenol Usage vs. Autism')
# plt.xlabel('Usage During Pregnancy')
# plt.ylabel('Hazard Ratio')

# sys.exit()


