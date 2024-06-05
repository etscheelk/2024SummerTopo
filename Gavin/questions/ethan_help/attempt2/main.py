import ctypes
import sys

# Example Python object
class Example:
    def __init__(self, value):
        self.value = value

example = Example(int("0xDEADBEEF69", 16))

example = int("0xDEADBEEF", 16)

# Get the memory address of the object
address = id(example)
# size = ctypes.sizeof(ctypes.py_object(example))
size = sys.getsizeof(example)

print(f"Address: {address}")
print(f"Size: {size}")

# print(sys.getsizeof(example))

# import ctypes

# Load the C shared library
lib = ctypes.CDLL('./read_memory.so')

# Define the function signature
lib.read_memory.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
lib.read_memory.restype = None

# Call the C function
lib.read_memory(ctypes.c_void_p(address), ctypes.c_size_t(size))