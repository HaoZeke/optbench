import numpy

force_calls = [1759947, 1149720, 876811, 2359419, 1315628, 1371225, 2643714, 1620370, 2063097]
unique_saddles = [115, 108, 98, 117, 108, 108, 134, 112, 118]
fraction_connected = [0.752053, 0.737566, 0.762945, 0.748346, 0.753225, 0.740408, 0.745586, 0.741735, 0.747634]
total_searches = [4872, 3197, 2472, 6501, 3643, 3779, 7307, 4507, 5706]

f = open('benchmark.dat','w')

f.write( 'force_calls ' + str(numpy.average(force_calls)) + '\n')
f.write( 'force_calls_mean_error ' + str(numpy.std(force_calls)) + '\n')
f.write( 'unique_saddles ' + str(numpy.average(unique_saddles)) + '\n')
f.write( 'fraction_connected ' + str(numpy.average(fraction_connected)) + '\n')
f.write( 'total_searches ' + str(numpy.average(total_searches)) + '\n')

f.close()
