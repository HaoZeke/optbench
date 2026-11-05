An OptKing start that asv stops at the class timeout is searched once.
The first track method records it as a failure with reason `timeout` and no force count.
The other track methods read that record.
Before, each of the six track methods ran the search again and met the same timeout.
One slow start therefore held its asv process for seven timeouts.
A killed search also no longer leaves an older record in place for the track methods to read.
