import numpy as np
import matplotlib.pyplot as plt
import random
import shapely
from robotFollowingRRT import robot
start = np.array([10.0, 10.0])

#Robot follower
#Init robot

fig, ax = plt.subplots(figsize = (7,7))

x=0
y=0
for i in range(5):
    ax.plot(x,y,'ro')
    x+=1
    y+=1
    plt.pause(0.10)
ax.plot(0.5,0.5,'yo')
plt.show()