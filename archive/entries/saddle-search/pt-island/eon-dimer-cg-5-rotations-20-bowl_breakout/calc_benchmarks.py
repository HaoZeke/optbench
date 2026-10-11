import numpy

force_calls = [4513755, 9239510, 3462588, 1982319, 2410466, 2834018, 2294515, 1363650, 3072485]
unique_saddles = [132, 159, 126, 111, 122, 128, 106, 106, 133]
fraction_connected = [0.692786, 0.705263, 0.701498, 0.709371, 0.705692, 0.694037, 0.702842, 0.712237, 0.710790]
total_searches = [6917, 14155, 5340, 3052, 3707, 4360, 3483, 2092, 4699]

f = open('benchmark.dat','w')

f.write( 'force_calls ' + str(numpy.average(force_calls)) + '\n')
f.write( 'force_calls_mean_error ' + str(numpy.std(force_calls)) + '\n')
f.write( 'unique_saddles ' + str(numpy.average(unique_saddles)) + '\n')
f.write( 'fraction_connected ' + str(numpy.average(fraction_connected)) + '\n')
f.write( 'total_searches ' + str(numpy.average(total_searches)) + '\n')

f.close()
