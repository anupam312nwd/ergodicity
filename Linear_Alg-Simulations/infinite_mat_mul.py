import numpy as np


def get_stochastic_mat(n=3):
    B = []
    for i in range(n):
        A = np.random.rand(n)
        A = A / sum(A)
        A = [np.round(a, 2) for a in A]
        A[n - 1] = round(1 - sum(A[: n - 1]), 2)
        B.append(A)
    B = np.array(B)
    return B


A = get_stochastic_mat()
B = get_stochastic_mat()
C = get_stochastic_mat()
print("matrix A -----")
print(A)
print("matrix B -----")
print(B)
# print(C)
print("-------------------")
print(np.round(np.matmul(A, B), 2))
