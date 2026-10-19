# This is statement is required by the build system to query build info
if __name__ == '__build__':
	raise Exception

def __set_attributes():
	global __date__, __version__, __build__

	import os.path

	__date__ = ' '.join('$Date: 2007-09-10 01:41:48 $'.split()[1:3])

	filename = os.path.join(os.path.dirname(__file__), 'version')
	__version__ = open(filename).read().strip()
	__build__ = int(__version__.split('.')[3])

__set_attributes()

__author__ = 'Daniel Sheppard <daniel.sheppard@mail.utexas.edu>\nLijun Xu <xulijun@mail.utexas.edu>\nWilliam Stier <wstier@u.washington.edu>\nRye Terrell <ryeterrell@ryeterrell.net>\nWenjie Tang <twj916@mail.utexas.edu>\nand Graeme Henkelman <henkelman@mail.utexas.edu>'

__doc__ = '''This is Gman.'''

__all__ = ["data","dyn","func","io","math","opt","parm","pot","min","neb"]
from . import data,dyn,func,io,math,opt,parm,pot,min,vzr
