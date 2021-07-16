import math

l = 3


def numerator(N):
    num = [0] * N
    for i in range(N):
        num[i] = math.comb(N, i) * pow(l, i) * i / (N - i)
    return sum(num)


N = 500
print(numerator(N) / pow(1 + l, N))
