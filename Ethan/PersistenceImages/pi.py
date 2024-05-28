import random

import matplotlib as plt
import plotly

arr = []

random.seed(1)

for _ in range(100):
    r1 = random.random()
    r2 = random.random()
    r2 = r2 * (1 - r1) + r1
    arr.append((r1,r2))

print(arr)


timePers = []

for p in arr:
    pers = p[1] - p[0]
    timePers.append((p[0], pers))
print(timePers)

plotly.plot(timePers, kind="scatter")

if __name__ == "__main__":
    
        
    pass