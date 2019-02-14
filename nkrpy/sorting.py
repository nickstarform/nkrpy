"""Various Sorting algorithms."""

# internal modules

# external modules
import numpy as np

# relative modules

# global attributes
__all__ = ('gravity', 'findLength', 'minv', 'maxv')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1



def gravity(obj):
    """Gravity Sort Method."""
    if all([type(x) == int and x >= 0 for x in obj]):
        original = list(obj)
        reference = [range(x) for x in obj]
    else:
        raise ValueError("All elements must be positive integers")
    intermediate = []
    index = 0
    previous = sum([1 for x in reference if len(x) > index])
    while previous:
        intermediate.append(range(previous))
        index += 1
        previous = sum([1 for x in reference if len(x) > index])
    index = 0
    previous = sum([1 for x in intermediate if len(x) > index])
    out = []
    while previous:
        out.append(previous)
        index += 1
        previous = sum([1 for x in intermediate if len(x) > index])
    out = out[::-1]
    return out

# Utility functions to find minimum
# and maximum of two elements


def minv(x, y):
    return x if(x < y) else y


def maxv(x, y):
    return x if(x > y) else y


# Returns length of the longest
# contiguous subarray. Assuming sorted


def findLength(arr, n=0):
    # n is the first n elements of the array to look in
    if n == 0:
        n = len(arr)
    # Initialize result
    max_len = 1
    i = 0
    vals = [0, 0]
    while i < len(list(range(n - 1))):
        # print('i:',i)

        # Initialize min and max for
        # all subarrays starting with i
        mn = arr[i]
        mx = arr[i]

        # Consider all subarrays starting
        # with i and ending with j
        j = i + 1
        while j < len(list(range(n - 1))):
            # print('j:',j)
            # Update min and max in
            # this subarray if needed
            mn = minv(mn, arr[j])
            mx = maxv(mx, arr[j])

            # If current subarray has
            # all contiguous elements
            if ((mx - mn) == (j - i)):
                # print('New Length')
                if max_len < (mx - mn + 1):
                    vals = [mn, mx]
                max_len = maxv(max_len, mx - mn + 1)
            else:
                # print('Keeping ',vals)
                i = j - 1
                j += 1
                break
            """
            print('i:{},j:{},mn:{},mx:{},max_len:{}'\
                  .format(i,j,mn,mx,max_len))
            """
            j += 1
        i += 1
    # returns length, [min,max],[indexes]
    return max_len, vals, [i for i, x in enumerate(arr)
                           if (vals[0] <= x <= vals[-1])]

# end of file
