"""Generated by the Chapel compiler."""

import atexit

from chplpitest.chplpitest import *

# Register cleanup function to be called at program exit.
atexit.register(chplpitest.chpl_cleanup)
