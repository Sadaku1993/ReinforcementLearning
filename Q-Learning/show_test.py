from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import  matplotlib.pyplot as plt
import math

size = 60
goal = np.array([50, 50])

x = np.arange(0, size)
y = np.arange(0, size)
X, Y = np.meshgrid(x, y)

Z = np.sqrt((X-goal[0])**2+(Y-goal[1])**2)

minz = Z.min()
maxz = Z.max()

Z /= (maxz - minz)

Z *= -1

Z += 1

print (x)
print (y)
print (X)
print (Y)
print (Z)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X,Y,Z)
plt.show()
