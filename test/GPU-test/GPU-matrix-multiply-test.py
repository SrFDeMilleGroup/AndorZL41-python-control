import time
import cupy as cp
import numpy as np
import jax.numpy as jnp
import jax

array_size = 4000
loop = 10

# numpy test
a = np.random.rand(array_size, array_size)
start = time.time()
for _ in range(loop):
    np.dot(a, a)
print("Time taken by numpy: ", time.time() - start)

# cupy test
b = cp.random.rand(array_size, array_size)
start = time.time()
for _ in range(loop):
    cp.dot(b, b)
print("Time taken by cupy: ", time.time() - start)
del b
cp._default_memory_pool.free_all_blocks()

# jax test
key = jax.random.PRNGKey(0)
c = jax.random.uniform(key, (array_size, array_size))
start = time.time()
for _ in range(loop):
    jnp.dot(c, c)
print("Time taken by jax: ", time.time() - start)
del c
cp._default_memory_pool.free_all_blocks()