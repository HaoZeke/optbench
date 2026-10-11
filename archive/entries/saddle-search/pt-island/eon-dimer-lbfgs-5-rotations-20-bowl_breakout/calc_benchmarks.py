import numpy

force_calls = [1299454, 1091820, 1819181, 856320, 1907947, 637068, 1664039, 1583382, 1249982]
unique_saddles = [132, 119, 138, 111, 142, 98, 149, 131, 116]
fraction_connected = [0.852782, 0.850216, 0.861961, 0.853946, 0.851744, 0.852779, 0.853945, 0.851108, 0.850596]
total_searches = [6453, 5321, 8954, 4245, 9234, 3077, 8086, 7717, 5957]

f = open('benchmark.dat','w')

f.write( 'force_calls ' + str(numpy.average(force_calls)) + '\n')
f.write( 'force_calls_mean_error ' + str(numpy.std(force_calls)) + '\n')
f.write( 'unique_saddles ' + str(numpy.average(unique_saddles)) + '\n')
f.write( 'fraction_connected ' + str(numpy.average(fraction_connected)) + '\n')
f.write( 'total_searches ' + str(numpy.average(total_searches)) + '\n')

f.close()
