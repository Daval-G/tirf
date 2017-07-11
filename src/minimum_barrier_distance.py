import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import numpy as np

def dist_fill(x, y, mins, image):
    if (np.sum(image[x, y]) == 0):
        return mins

    next = [[[x+1, y],[x, y]],[[x-1, y],[x, y]],[[x, y+1],[x, y]],[[x, y-1],[x, y]]]
    mins[x, y] = 0
    count = 0

    while (next != []):
        nx, ny = next[0][0]
        px, py = next[0][1]
        
        next = next[1:]
        if (nx < 0 or ny < 0 or nx >= image.shape[0] or ny >= image.shape[1]):
            continue

        diff = np.sum(np.abs(image[nx, ny] - image[px, py]))
        val = mins[px, py] + diff

        if (val < mins[nx, ny]):
            mins[nx, ny] = val
            next.append([[nx + 1, ny], [nx, ny]])
            next.append([[nx - 1, ny], [nx, ny]])
            next.append([[nx, ny + 1], [nx, ny]])
            next.append([[nx, ny - 1], [nx, ny]])

        count += 1
    return mins

import PIL
from PIL import Image

im = Image.open('./test/quasi_facile/20160525_143754.jpg')
tau = float(im.size[0]) / im.size[1]
im = im.resize((int(300 * tau), 300), PIL.Image.ANTIALIAS)
im.save('./test/reduced.jpg')

im = imread("./test/reduced.jpg")
mins = 100000 * 255 * np.ones((im.shape[0], im.shape[1]))

for i in range(im.shape[1]):
    mins = dist_fill(0, i, mins, im)
    mins = dist_fill(im.shape[0] - 1, i, mins, im)
plt.imshow(mins, cmap = plt.get_cmap('gray'))
plt.show()

for i in range(im.shape[0]):
    mins = dist_fill(i, 0, mins, im)
    mins = dist_fill(i, im.shape[1] - 1, mins, im)
plt.imshow(mins, cmap = plt.get_cmap('gray'))
plt.show()

imsave("./test/result.jpg", mins)

maxi = np.max(mins)
limi = maxi // 4
for i in range(mins.shape[0]):
    for j in range(mins.shape[1]):
        mins[i, j] = (mins[i, j] // limi)
plt.imshow(mins, cmap = plt.get_cmap('gray'))
plt.show()