import tqdm
import numpy as np
triplets_dl = np.zeros((1500, 3))
for a,b, c in tqdm.tqdm(triplets_dl, ncols=100):
    print(a, b, c)