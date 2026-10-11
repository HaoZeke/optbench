import numpy

force_calls = [2650346, 2798870, 1618008, 5325768, 2631100, 2257649, 1307105, 3890156, 2111780]
unique_saddles = [104, 102, 91, 120, 107, 101, 81, 111, 95]
fraction_connected = [0.791149, 0.790380, 0.788537, 0.804323, 0.795573, 0.802943, 0.801640, 0.797938, 0.799295]
total_searches = [7570, 7900, 4606, 15173, 7455, 6455, 3781, 11056, 5954]

f = open('benchmark.dat','w')

f.write( 'force_calls ' + str(numpy.average(force_calls)) + '\n')
f.write( 'force_calls_mean_error ' + str(numpy.std(force_calls)) + '\n')
f.write( 'unique_saddles ' + str(numpy.average(unique_saddles)) + '\n')
f.write( 'fraction_connected ' + str(numpy.average(fraction_connected)) + '\n')
f.write( 'total_searches ' + str(numpy.average(total_searches)) + '\n')

f.close()
