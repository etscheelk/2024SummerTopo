from libc.stdint cimport *
from chplrt cimport *

cdef extern from "chplpitest.h":
	void chpl__init_image(int64_t _ln, int32_t _fn);
	void chpl__init_pi(int64_t _ln, int32_t _fn);
