import os
import itertools

def log(print_str, fn):
    with open(fn, "a+") as f:
        f.write(print_str)
        f.write("\n")