import numpy as np

def labelToArray(label, n):
    vec = np.zeros(n)
    l = len(label)
    for i in range(int(l/2)):
        indices = range(label[2 * i], label[2 * i] + label[2 * i + 1])
        for j in indices:
            vec[j] = 1
    return vec

a = labelToArray([0,2,4,1], 5)
print(a)
