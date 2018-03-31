import numpy as np

## Example solution of softmax implementation

def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=0)

scores = [[[0.1, 0.2, 0.3, 6],
           [22, 14, 0.5, 1.6],
           [3, 8, 7, 6]],
          [[1, 2, 3, 6],
           [2, 4, 5, 6],
           [3, 8, 7, 6]],
          [[3, 1, 1, 1],
           [2, 4, 5, 6],
           [3, 8, 7, 6]]]


print(softmax(scores))