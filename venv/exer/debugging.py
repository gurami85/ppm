import matplotlib.pyplot as plt
import numpy as np

# initialize empty arrays
array_len = 10000   # desired length of arrays
var1 = np.zeros(array_len)
var2 = np.zeros(array_len)
var3 = np.zeros(array_len)
var4 = np.zeros(array_len)

# define two functions:
def function_1(input):
    # function 1
    for i in range(array_len):
        var1[i] = 2 + i*input
        var2[i] = 2*var1[i]
    return var1, var2

def function_2(input):
    # function 2
    for j in range(array_len):
        var3[j] = 3 + j*input
        var4[j] = 2*var3[j]
    return var3, var4

# add var1 and var3, and plot result:
output = function_1(1)[0] + function_2(2)[0]
plt.plot(output)
plt.show(output.all())