#!/usr/bin/env python3

import numpy as np
import datetime

from lowest_eigenvector import run_all

# CHANGE THIS: Number of processors to run tests with.
nprocs = int(input())

# Finite difference step size
eta = 1e-6

# Convergence threshold for lowest eigenvector search
vreftol = 0.99

output = run_all(nprocs, eta, "gradient", "projection", vreftol)
results = np.array(output)[:, 0]
mean_best = np.average(results)
minimum_best = np.min(results)
maximum_best = np.max(results)
bench = open('benchmark.dat','w')
bench.write('force_calls ' + '{:.0f}'.format(mean_best) + '\n')
bench.write('force_calls_min ' + str(minimum_best) + '\n')
bench.write('force_calls_max ' + str(maximum_best) + '\n')
bench.write('nfailed 0\n')
bench.write('code Sella\ncode_version 0.1.1\ndate ' + str(datetime.datetime.today()).split()[0] + '\ncontributor Eric D. Hermes, Khachik Sargsyan, Habib Najm, Judit Zádor')
'''
print('Table 3:')
print('code  |  mean  |   min  |   max ')
print('------+--------+--------+-------')
print('Sella | {:6} | {:6} | {:6}'
      ''.format(mean_best, minimum_best, maximum_best))
print('Optim |     25 |     13 |     58')
print('Pele  |     25 |     12 |     61')
print('')

print('Table 4:')
print('x0       | zero-curvature modes |  mean  |   min  |   max ')
print('---------+----------------------+--------+--------+-------')
for v0_guess in ["optbench", "gradient"]:
    for zero in ["projection", "shifting"]:
        if v0_guess == 'gradient' and zero == 'projection':
            mean = mean_best
            minimum = minimum_best
            maximum = maximum_best
        else:
            output = run_all(nprocs, eta, v0_guess, zero, vreftol)
            results = np.array(output)[:, 0]
            mean = np.average(results)
            minimum = np.min(results)
            maximum = np.max(results)
        print('{:<8} | {:<20} | {:6} | {:6} | {:6}'
              ''.format(v0_guess, zero, mean, minimum, maximum))
'''
