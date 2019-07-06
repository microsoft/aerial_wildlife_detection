'''
    Miscellaneous helper functions.

    2019 Benjamin Kellenberger
'''

def array_split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr = arr[size:]
     arrs.append(arr)
     return arrs