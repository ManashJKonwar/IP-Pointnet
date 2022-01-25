__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

# Function to find starting and end index
def find_index (a, n, key):
    start = -1
    
    # Traverse from beginning to find
    # first occurrence
    for i in range(n):
        if a[i] == key:
            start = i
            break

    if start == -1:
        print("Key not present in array")
        return None, None
    
    # Traverse from end to find last
    # occurrence.
    end = start
    for i in range(n-1, start - 1, -1):
        if a[i] == key:
            end = i
            break
    if start == end:
        # print("Only one key is present at index : ", start)
        return start, end
    else:
        # print("Start index: ", start)
        # print("Last index: ", end)
        return start, end
