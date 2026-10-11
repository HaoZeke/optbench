#!/usr/bin/env python
from sys import exit

f = open('states/0/search_results.txt')
f.readline()
f.readline()
nfound = 0
ntofind = 27
cutoff = 1.5
force_calls = 0
total_found = 0
connected = 0
total_searches = 0
not_connected = 0
for line in f:
    total_searches += 1
    fields = line.split()
    barrier = float(fields[2])
    force_calls += int(fields[4]) + int(fields[5])
    result = fields[-1]
    if 'good' in result: total_found += 1
    if 'Connected' in result: not_connected += 1
    if 'good' in result and barrier < cutoff:
        nfound += 1
        if nfound >= ntofind:
            print 'force_calls %i' % force_calls
            print 'unique_saddles %i' % total_found
            print 'fraction_connected %f' % (float(total_searches-not_connected)/float(total_searches))
            print 'total_searches %i' % total_searches
            exit(0)
print 'found %i/%i %i force calls' % (nfound,ntofind, force_calls)
exit(1)
