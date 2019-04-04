from time import time
import numpy as np

n = 40000
a = np.random.rand(n)
b = np.random.rand(n)

start = time()
c = a * b
print(f'[BROADCAST] Time: {time() - start:.10f} sec')

start = time()
d = []
for i in range(n):
    d.append(a[i] * b[i])
d = np.array(d)
print(f'     [LOOP] Time: {time() - start:.10f} sec')

print(f'Czy są takie same: {np.all(c == d)}')