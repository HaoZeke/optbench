#!/usr/bin/env python
import numpy as np
import sys, copy
from sys import argv
from os.path import join

#path = argv[1]
path = './states/0/'

rate_accuracy = 0.999
#all the barriers below 1.5 eV
#barrier_list = [0.601, 0.601, 0.620, 0.986, 0.987, 0.989, 0.986, 0.987, 0.989, 1.196, 1.196, 1.196, 1.196, 1.207, 1.207, 1.479, 1.480, 1.480, 1.481, 1.481, 1.483, 1.483, 1.491, 1.491, 1.493, 1.493, 1.493, 1.493]
#all the barriers below 1.3 eV
barrier_list = [0.601, 0.601, 0.620, 0.986, 0.987, 0.989, 0.986, 0.987, 0.989, 1.196, 1.196, 1.196, 1.196, 1.207, 1.207]
barrier_list0 = copy.copy(barrier_list)
print barrier_list
print len(barrier_list)
rate_ref = 155055901.244

f = open(join(path, 'processtable'))
processes = {}
test_list = []
last_barrier = -1
last_rate    = -1
rate_sum = 0
barrier_ids_ref = []
for line in f:
    fields = line.split()
    if fields[0] == 'proc': continue
    number  = int(fields[0])
    barrier = float(fields[6])
    barrier = round(barrier, 3)
    rate    = float(fields[7])
    frequency = int(fields[8])+1

    processes[number] = {'barrier':barrier, 'frequency':frequency}

    if barrier < 1.3 and barrier not in barrier_list0: 
        print "Found an unexped barrier, check it carefully:"
        print barrier
        #sys.exit()
    if barrier < 1.3:
        test_list.append(barrier)
    if barrier in barrier_list:
        barrier_ids_ref.append(number)
        i = barrier_list.index(barrier)
        del barrier_list[i]
    if len(barrier_list) == 0 and last_barrier < 0: 
        last_barrier = number

    rate_sum += rate 
    if rate_sum >= rate_ref * rate_accuracy and last_rate < 0:
        last_rate    = number

#print rate_sum
test_list.sort()
print test_list
print len(test_list)

f.close()
if last_barrier > 0:
    print "Found all the barriers less than 1.5eV."
else: 
    print "Failed, processes with the following barriers are missed:"
    print barrier_list
    print "Please run more searches."

if last_rate > 0:
    print "The escape rate reaches ", rate_accuracy, "of the reference hTST rate."
else: 
    print "Failed, the escape rate did not reach the confidence of ", rate_accuracy
    print "Please run more searches."

f = open(join(path, 'search_results.txt'))
Temp = 700
prefactor = 1.2e12
kb   = 8.6173324e-5
f.readline()
f.readline()
fcs_bsum    = 0
search_bnum = 0
fcs_rsum    = 0
search_rnum = 0
rate_sumt   = 0.0
fcsbarrier_sum     = []
fcsrate_sum        = []
searchbarrier_num  = []
searchrate_num     = []
barrier_ids  = []
rate_ids     = []
for line in f:
    search_bnum +=1
    search_rnum +=1
    fields = line.split()
    saddle_forcecalls = int(fields[4])
    barrier   = float(fields[2])
    fcs_bsum += saddle_forcecalls
    fcs_rsum += saddle_forcecalls

    result_string = fields[-1]
    if '-' not in result_string:
        continue
    first_part = result_string.split('-')[0]
    if first_part not in ("good", "repeat"):
        continue

    id = int(result_string.split('-')[1])
    if id in barrier_ids_ref and id not in barrier_ids:
        barrier_ids.append(id)
    if len(barrier_ids) == len(barrier_ids_ref):
        barrier_ids = []
        searchbarrier_num.append(search_bnum)
        fcsbarrier_sum.append(fcs_bsum)
        search_bnum = 0
        fcs_bsum    = 0

    if rate_sumt < rate_ref * rate_accuracy and id not in rate_ids:
        rate_ids.append(id)
        rate_sumt += prefactor * np.exp(-barrier / (kb * Temp))

    if rate_sumt >= rate_ref * rate_accuracy:
        rate_ids    = []
        searchrate_num.append(search_rnum)
        fcsrate_sum.append(fcs_rsum)
        search_rnum = 0
        fcs_rsum    = 0
        rate_sumt   = 0.0

print "Number of Barrier Searches:"
print searchbarrier_num
print "ForceCalls:"
print fcsbarrier_sum

#print "Number of Rate Searches:"
#print searchrate_num 
#print "ForceCalls:"
#print fcsrate_sum

barrier_jobs = np.mean(searchbarrier_num)
barrier_fcs  = np.mean(fcsbarrier_sum)
rate_jobs  = np.mean(searchrate_num)
rate_fcs   = np.mean(fcsrate_sum)

from datetime import date
d     = date.today()
today = d.strftime("%d %b %Y")
resultfile = open('benchmark.dat','w')
resultfile.write("barrier_jobs %i \n" % barrier_jobs)
resultfile.write("barrier_force_calls %.3e \n" % barrier_fcs)
resultfile.write("average over %i \n" % len(searchbarrier_num))
resultfile.write("rate_jobs %i \n" % rate_jobs)
resultfile.write("rate_force_calls % .3e \n" % rate_fcs)
resultfile.write("average over %i \n" % len(searchrate_num))
resultfile.write("algorithm Dimer-LBFGS rotation: max=50, tol=1\n")
resultfile.write("code Eon r2006\n")
resultfile.write("date %s\n" % today)
resultfile.write("contributor Penghao Xiao\n")
resultfile.close()


