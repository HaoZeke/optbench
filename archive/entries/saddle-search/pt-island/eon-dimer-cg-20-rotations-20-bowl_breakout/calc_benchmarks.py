import numpy

force_calls = [3692020, 5754515, 6761579, 4444445, 10296849, 5764608, 5114860, 10850004, 11889043]
unique_saddles = [115, 120, 121, 120, 135, 120, 109, 135, 134]
fraction_connected = [0.745130, 0.748445, 0.743597, 0.748568, 0.753682, 0.745764, 0.744709, 0.746776, 0.746234]
total_searches = [6109, 9485, 11166, 7334, 17108, 9503, 8457, 17834, 19652]

f = open('benchmark.dat','w')

f.write( 'force_calls ' + str(numpy.average(force_calls)) + '\n')
f.write( 'force_calls_mean_error ' + str(numpy.std(force_calls)) + '\n')
f.write( 'unique_saddles ' + str(numpy.average(unique_saddles)) + '\n')
f.write( 'fraction_connected ' + str(numpy.average(fraction_connected)) + '\n')
f.write( 'total_searches ' + str(numpy.average(total_searches)) + '\n')

f.close()
