from anms.ssc import square_covering_adaptive_nms
import time
import numpy as np

N = 10000
width, height = 1280, 835

kpts_x = np.random.randint(width, size=(N,))
kpts_y = np.random.randint(height, size=(N,))
kpts = np.stack([kpts_x, kpts_y], axis=-1)
resp = np.random.random(N)

print(kpts.shape, resp.shape)

n_repeats = 1
start = time.time()
for _ in range(n_repeats):
    x = square_covering_adaptive_nms(kpts, resp, width=1280, height=835, target_num_kpts=2048, indices_only=True)
print(time.time() - start)

print(x.dtype)
