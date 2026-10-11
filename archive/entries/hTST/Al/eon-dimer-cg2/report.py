#!/usr/bin/env python
import numpy as np
import sys, copy
from sys import argv
from os.path import join

#path = argv[1]
path = './states/0/'

rate_accuracy = 0.999
#all the barriers below 0.5 eV
barrier_list = [0.233, 0.233, 0.233, 0.233, 0.372, 0.372, 0.372, 0.372]
#all the barriers below 0.4 eV
barrier_list0 = copy.copy(barrier_list)
print barrier_list
print len(barrier_list)
rate_ref = 592041917.0

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

    if barrier < 0.4 and barrier not in barrier_list0: 
        print "Found an unexped barrier, check it carefully:"
        print barrier
        #sys.exit()
    if barrier < 0.4:
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

print rate_sum
test_list.sort()
print test_list
print len(test_list)

f.close()
if last_barrier > 0:
    print "Found all the barriers less than 0.4eV. Total number of processes examined:", last_barrier
else: 
    print "Failed, processes with the following barriers are missed:"
    print barrier_list
    print "Please run more searches."

if last_rate > 0:
    print "The escape rate reaches ", rate_accuracy, "of the reference hTST rate. Total number of processes examined:"
    print last_rate
else: 
    print "Failed, the escape rate did not reach the confidence of ", rate_accuracy
    print "Please run more searches."

f = open(join(path, 'search_results.txt'))
Temp = 300
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

#print "Number of Barrier Searches:"
#print searchbarrier_num
#print "ForceCalls:"
#print fcsbarrier_sum

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
resultfile.write("algorithm Dimer-CG rotation: max=10 tol=1\n")
resultfile.write("code Eon\n")
resultfile.write("code r2006\n")
resultfile.write("date %s\n" % today)
resultfile.write("contributor Penghao Xiao\n")
resultfile.close()

