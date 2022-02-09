import numpy as np
from deepdish.io import load

from ssc import square_covering_adaptive_nms

import time

kpts = load('/home/ostap/trash/kpts.h5')
resp = load('/home/ostap/trash/responses.h5')

print(kpts.shape, resp.shape)

n_repeats = 1
start = time.time()
for _ in range(n_repeats):
    x = square_covering_adaptive_nms(kpts, resp, width=1280, height=835, target_num_kpts=2048)
print(time.time() - start)

print(x.shape, x.dtype)