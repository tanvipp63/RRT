import numpy as np
import matplotlib.pyplot as plt
import random
import shapely
from robotFollowingRRT import robot
start = np.array([10.0, 10.0])

#Robot follower
#Init robot
mu = np.array([[start[0]], [start[1]], [0]])
robot = robot(mu)
robot = shapely.Polygon(robot.getCorners())
x1, y1 = robot.exterior.xy
plt.plot(x1, y1)
plt.grid()
plt.show()