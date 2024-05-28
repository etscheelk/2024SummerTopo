#!python
#cython: language_level=3
from chplrt cimport chpl_library_init, chpl_library_finalize, chpl_external_array, chpl_make_external_array, chpl_make_external_array_ptr, chpl_free_external_array, chpl_opaque_array, cleanupOpaqueArray, chpl_free, chpl_byte_buffer, chpl_byte_buffer_free, PyBytes_FromStringAndSize
from chpl_chplpitest cimport chpl__init_image, chpl__init_pi

import numpy
cimport numpy
import ctypes
from libc.stdint cimport intptr_t

_chpl_cleanup_callback = None

def chpl_set_cleanup_callback(callback):
	global _chpl_cleanup_callback
	_chpl_cleanup_callback = callback

def chpl_setup():
	cdef char** args = ['chplpitest']
	chpl_library_init(1, args)
	chpl__init_image(1, 1)
	chpl__init_pi(1, 1)

def chpl_cleanup():
	global _chpl_cleanup_callback
	callback = _chpl_cleanup_callback
	if not callback is None:
		callback()
	chpl_library_finalize()

cdef class ChplOpaqueArray:
	cdef chpl_opaque_array val

	cdef inline setVal(self, chpl_opaque_array val):
		self.val = val

	def cleanup(self):
		cleanupOpaqueArray(&self.val)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.cleanup()

