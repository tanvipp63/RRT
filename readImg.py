from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

#Generate occupancy grid to implement RRT in
img = Image.open('cspace3.png')

img = ImageOps.grayscale(img)

# Resize the image to the desired dimensions (e.g., 100x100)
desired_width, desired_height = 258, 250
img.thumbnail((desired_width, desired_height), Image.BILINEAR)


np_img = np.array(img)
np_img = ~np_img #inverts B&W
np_img[np_img > 0] = 1
plt.set_cmap('binary')
plt.imshow(np_img)

np.save('cspace.npy', np_img)

grid = np.load('cspace.npy')
plt.imshow(grid)
plt.tight_layout()
plt.show()