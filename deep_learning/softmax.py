import numpy as np

a = np.random.randn(5)
expa = np.exp(a)

answer = expa / expa.sum()

print(answer)
print(answer.sum())

A = np.random.randn(100, 5)

expA = np.exp(A)

answer = expA / expA.sum(axis=1, keepdims=True) # important line, take note of the params passed!

print(answer)
print(answer.sum(axis=1))