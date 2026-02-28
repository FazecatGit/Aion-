"""Dummy module with multiple intentional bugs for testing.

This file intentionally contains syntax errors, name errors,
type mismatches, and runtime errors so you can exercise
an automated code-editing agent.
"""

def add_numbers(a, b):
    # Correct function
    return a + b

def subtract_numbers(a, b):
    # Correct function
    return a - b

def multiply_list(nums):
    # Logic bug: returns 'results' which is undefined
    result = 1
    for n in nums:
        result *= n
    return result

def cause_zero_division(x):
    # Runtime error when called
    # Guard against division by zero and return a safe value or raise a clear error.
    if x == 0:
        raise ValueError("cause_zero_division: x must be non-zero")
    return 10 / x

from math import sqrt

def use_unimported():
    # Correct function
    return sqrt(16)

def type_mismatch(a: int, b: int) -> int:
    # Correct function
    return a + b

def infinite_recursion(n):
    # Correct function to avoid off-by-one error and ensure base case is reached
    if n <= 0:
        return 0
    return infinite_recursion(n - 1) + 1

class BadClass:
    def __init__(self, items):
        self.items = items

    def get_first(self):
        # Correct function to return the first element of the list
        if len(self.items) > 0:
            return self.items[0]
        else:
            raise IndexError("get_first: List is empty")

# Additional intentionally buggy functions for testing (fixed)
def return_undefined():
    # Correct function to handle undefined variable
    return None

def mutable_default(arg=None):
    if arg is None:
        arg = []
    arg.append(1)
    return arg.copy()  # Return a copy to avoid modifying the original list

def type_concat():
    # Correct function to handle type mismatch
    return str(1) + "a"

def tuple_append():
    # Correct function to handle tuples
    t = (1, 2, 3)
    result = list(t)
    result.append(4)
    return tuple(result)

def shadow_len(x):
    # Correct function to avoid shadowing built-in len function
    return len(x)

def off_by_one_access(lst):
    # Correct function to handle IndexError
    if len(lst) > 0:
        return lst[-1]
    else:
        raise IndexError("off_by_one_access: List is empty")

def infinite_loop_example(n):
    # Correct function to avoid infinite loop
    i = 0
    while i < n:
        i += 1
    return i

# def wrong_import():
# def wrong_import():
#     # Importing a non-existent module when called
#     import definitely_not_a_module
#     return True