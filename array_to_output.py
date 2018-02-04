import numpy as np
def array_to_output(array):
    array = np.split(array, array.size / 2)
    output = np.array([])
    for item in array:
        index = item[0]
        number = item[1]
        while output.size + 1 <= index:
            output = np.append(output, 0)
        for i in range(0, number):
            output = np.append(output, 1)
    return output

output = array_to_output(np.array([2,1,4,2,10,4]))
print(output)