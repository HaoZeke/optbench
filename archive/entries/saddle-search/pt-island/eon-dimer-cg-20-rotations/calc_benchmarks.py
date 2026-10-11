import numpy

force_calls = [6856981, 5519987, 8709751, 6094162, 6725677, 14882934, 7856482, 6331587, 8546070]
unique_saddles = [114, 120, 131, 117, 116, 140, 121, 112, 126]
fraction_connected = [0.631166, 0.637438, 0.637731, 0.627427, 0.639895, 0.635647, 0.630145, 0.632750, 0.635653]
total_searches = [10845, 8804, 13893, 9684, 10683, 23587, 12440, 10064, 13564]

f = open('benchmark.dat','w')

f.write( 'force_calls ' + str(numpy.average(force_calls)) + '\n')
f.write( 'force_calls_mean_error ' + str(numpy.std(force_calls)) + '\n')
f.write( 'unique_saddles ' + str(numpy.average(unique_saddles)) + '\n')
f.write( 'fraction_connected ' + str(numpy.average(fraction_connected)) + '\n')
f.write( 'total_searches ' + str(numpy.average(total_searches)) + '\n')

f.close()
