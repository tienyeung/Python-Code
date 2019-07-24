import numpy
def quick(arr):
  if len(arr)<2:
    return arr
  else:
    std=arr[0]
    small = [i for i in arr[1:] if i <= std]
    big = [i for i in arr[1:] if i > std]
    return quick(small)+[std]+quick(big)


print(quick([1,3,2]))