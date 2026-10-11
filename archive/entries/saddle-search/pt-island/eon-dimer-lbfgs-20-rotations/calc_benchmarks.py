import numpy

force_calls = [3608088, 2557765, 7282827, 2829418, 1853628, 1589484, 4647380, 2894413, 2810389]
unique_saddles = [112, 105, 140, 101, 94, 88, 121, 110, 101]
fraction_connected = [0.745190, 0.747022, 0.746180, 0.746944, 0.742484, 0.752052, 0.739946, 0.747297, 0.745212]
total_searches = [10447, 7388, 21074, 8263, 5355, 4630, 13328, 8417, 8093]

f = open('benchmark.dat','w')

f.write( 'force_calls ' + str(numpy.average(force_calls)) + '\n')
f.write( 'force_calls_mean_error ' + str(numpy.std(force_calls)) + '\n')
f.write( 'unique_saddles ' + str(numpy.average(unique_saddles)) + '\n')
f.write( 'fraction_connected ' + str(numpy.average(fraction_connected)) + '\n')
f.write( 'total_searches ' + str(numpy.average(total_searches)) + '\n')

f.close()
