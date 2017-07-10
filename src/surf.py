import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import numpy as np
import numpy.linalg as la
import math
from math import sqrt

#------------------------#
#-----SURF ALGORITHM-----#
#------------------------#
image = imread('test/facile/20160525_143739.jpg')

#-----PREPROCESSING-----#
# GRAYSCALE
image = np.dot(image, np.array([0.241,0.691,0.068]))
#plt.imshow(image, cmap = plt.get_cmap('gray'))
#plt.show()

# BORDER CONDITIONS

# INTERPOLATION (???)

#-----FEATURE FILTERING-----#
# INTEGRAL IMAGE
integral = np.cumsum(np.cumsum(image, axis=0), axis=1)

# FIRST ORDER BOX FILTER
def first_order_x_image(L, integral):
    return first_order_y_image(L, integral.T).T

def first_order_y_image(L, i):
    l = int(0.8 * L)
    result = (np.roll(i, (-l, -l), axis=(0,1)) + np.roll(i, (l + 1, 0), axis=(0,1)) - np.roll(i, (-l, 0), axis=(0,1)) - np.roll(i, (l + 1, -l), axis=(0,1))) \
           - (np.roll(i, (-l, 1), axis=(0,1)) + np.roll(i, (l + 1, l + 1), axis=(0,1)) - np.roll(i, (-l, l + 1), axis=(0,1)) - np.roll(i, (l + 1, 1), axis=(0,1)))
    result[:l + 1,:] = 0
    result[result.shape[0] - l:,:] = 0
    result[:,:l + 1] = 0
    result[:,result.shape[1] - l:] = 0
    return result

#L = 5
#plt.imshow(first_order_x_image(L, integral), cmap = plt.get_cmap('gray'))
#plt.show()
#plt.imshow(first_order_y_image(L, integral), cmap = plt.get_cmap('gray'))
#plt.show()

# SECOND ORDER BOX FILTER
def second_order_xy_image(L, i):
    result = (np.roll(i, (-L, -L), axis=(0,1)) + i)                                     \
           + (np.roll(i, (1, 1), axis=(0,1)) + np.roll(i, (L + 1, L + 1), axis=(0,1)))  \
           + (np.roll(i, (1, 0), axis=(0,1)) + np.roll(i, (L + 1, -L), axis=(0,1)))     \
           + (np.roll(i, (0, 1), axis=(0,1)) + np.roll(i, (-L, L + 1), axis=(0,1)))     \
           - (np.roll(i, (0, -L), axis=(0,1)) + np.roll(i, (-L, 0), axis=(0,1)))        \
           - (np.roll(i, (1, L + 1), axis=(0,1)) + np.roll(i, (L + 1, 1), axis=(0,1)))  \
           - (np.roll(i, (1, -L), axis=(0,1)) + np.roll(i, (L + 1, 0), axis=(0,1)))     \
           - (np.roll(i, (-L, 1), axis=(0,1)) + np.roll(i, (0, L + 1), axis=(0,1)))
    result[:L + 1,:] = 0
    result[result.shape[0] - L:,:] = 0
    result[:,:L + 1] = 0
    result[:,result.shape[1] - L:] = 0
    return result

def second_order_xx_image(L, integral):
    return second_order_yy_image(L, integral.T).T

def second_order_yy_image(L, i):
    L_2 = L // 2
    L3_2 = 3 * L_2
    result = (np.roll(i, (-L, -L3_2), axis=(0,1)) + np.roll(i, (L + 1, L3_2 + 1), axis=(0,1)))      \
           + (np.roll(i, (L + 1, -L_2), axis=(0,1)) + np.roll(i, (-L, L_2 + 1), axis=(0,1))) * 3    \
           - (np.roll(i, (-L, -L_2), axis=(0,1)) + np.roll(i, (L + 1, L_2 + 1), axis=(0,1))) * 3    \
           - (np.roll(i, (L + 1, -L3_2), axis=(0,1)) + np.roll(i, (-L, L3_2 + 1), axis=(0,1)))
    result[:L3_2 + 2,:] = 0
    result[result.shape[0] - L3_2 - 2:,:] = 0
    result[:,:L3_2 + 2] = 0
    result[:,result.shape[1] - L3_2 - 2:] = 0
    return result

#L = 5
#plt.imshow(second_order_xy_image(L, integral), cmap = plt.get_cmap('gray'))
#plt.show()
#plt.imshow(second_order_xx_image(L, integral), cmap = plt.get_cmap('gray'))
#plt.show()
#plt.imshow(second_order_yy_image(L, integral), cmap = plt.get_cmap('gray'))
#plt.show()

# DETERMINANT OF HESSIAN
def determinant_of_hessian_image(L, integral):
    Dxx = second_order_xx_image(L, integral)
    Dyy = second_order_yy_image(L, integral)
    Dxy = second_order_xy_image(L, integral)
    
    w = sqrt((2 * L - 1) / (2 * L))
    result = (np.multiply(Dxx, Dyy) - (w * Dxy) ** 2) / (L ** 4)

    return result

# SUBSAMPLING (2^(o-1))

#-----FEATURE SELECTION-----#
def non_maximum_suppression(x, y, hessian):
    return (hessian[x, y] >= hessian[x + 1, y] and hessian[x, y] >= hessian[x - 1, y]) \
        or (hessian[x, y] >= hessian[x, y + 1] and hessian[x, y] >= hessian[x, y - 1]) \
        or (hessian[x, y] >= hessian[x + 1, y + 1] and hessian[x, y] >= hessian[x - 1, y - 1]) \
        or (hessian[x, y] >= hessian[x + 1, y - 1] and hessian[x, y] >= hessian[x - 1, y + 1])

def location_refinement(x, y, L, hessian, prev_hessian, next_hessian):    
    dx = (hessian[x + 1, y] - hessian[x - 1, y]) // 2
    dy = (hessian[x, y + 1] - hessian[x, y - 1]) // 2
    dL =  (next_hessian[x, y] - prev_hessian[x, y]) // 4
    d0 = np.array([[dx], [dy], [dL]])
    
    Hxx = (hessian[x + 1, y] + hessian[x - 1, y] - 2 * hessian[x, y])
    Hyy = (hessian[x, y + 1] + hessian[x, y - 1] - 2 * hessian[x, y])
    Hxy = (hessian[x + 1, y + 1] + hessian[x - 1, y - 10] - hessian[x - 1, y + 1] - hessian[x + 1, y - 1]) // 4
    HxL = (next_hessian[x + 1, y] + prev_hessian[x - 1, y] - next_hessian[x - 1, y] - prev_hessian[x + 1, y]) // 8
    HyL = (next_hessian[x, y + 1] + prev_hessian[x, y - 1] - next_hessian[x, y - 1] - prev_hessian[x, y + 1]) // 8
    HLL = (next_hessian[x, y] + prev_hessian[x, y] - hessian[x, y]) // 4
    H0  = np.array([[Hxx, Hxy, HxL],[Hxy, Hyy, HyL],[HxL, HyL, HLL]])
    E   = - np.dot(np.linalg.inv(H0), d0)
    if (np.array([abs(E[0]), abs(E[1]), abs(E[2]) // 2]).max() < 1):
        return np.array([[x], [y], [L]]) + E
    else:
        return None

def feature_selection(hessian, prev_hessian, next_hessian, L):
    thres = 8    * np.max(hessian) // 10
    features = []
    for x in range(hessian.shape[0]):
        for y in range(hessian.shape[1]):
            if (hessian[x, y]) > thres:
                if non_maximum_suppression(x, y, hessian):
                    pt = location_refinement(x, y, L, hessian, prev_hessian, next_hessian)
                    if (not (pt is None)) and (not ((int(pt[0]), int(pt[1]), int(pt[2])) in features)):
                        features.append((int(pt[0]), int(pt[1]), int(pt[2])))
    return features

#im3 = determinant_of_hessian_image(3, integral)
#im5 = determinant_of_hessian_image(5, integral)
#im7 = determinant_of_hessian_image(7, integral)
#im9 = determinant_of_hessian_image(9, integral)
#features5 = feature_selection(im5, im3, im7, 5)
#print(len(features5))
#features7 = feature_selection(im7, im5, im9, 7)
#print(len(features7))

#-----INTEREST POINT DESCRIPTION-----#
def linear_scale_space(sigma, k):
    result = 1.0 * np.arange(-k, k + 1)
    result = - (result ** 2) / (2 * (sigma ** 2))
    result = np.exp(result)

    result = np.outer(result, result)
    result = result / np.sum(result)
    
    return result

def py_ang(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def orientation(x, y, L, Dx, Dy):
    sigma   = int(0.4 * L)
    gauss   = linear_scale_space(1, 3)
    phis    = np.zeros((13,13, 2))
    for dx in range(13):
        dx6 = dx - 6
        for dy in range(13):
            dy6 = dy - 6
            if ((dx6) ** 2 + (dy6) ** 2 <= 36):
                phis[dx, dy] = gauss[dx // 2, dy // 2] * np.array([Dx[x + sigma * dx6, y + sigma * dy6], Dy[x + sigma * dx6, y + sigma * dy6]])

    norm    = la.norm(phis, axis=2)
    angl    = np.arctan2(np.cross(phis, [1,0]), np.dot(phis, [1,0]))
    phik    = [np.sum(np.where(np.abs(angl - (k * math.pi / 20)) < np.pi / 6, norm, 0)) for k in range(40)]
    where   = np.abs(angl - (np.argmax(phik) * math.pi / 20)) < (np.pi / 6)
    return py_ang([np.sum(np.where(where, phis[:,:,0], 0)), np.sum(np.where(where, phis[:,:,1], 0))], [1,0])

#x   = image.shape[0] // 2
#y   = image.shape[1] // 2
#L   = 3
#Dx  = first_order_x_image(L, integral)
#Dy  = first_order_y_image(L, integral)
#
#e = 1
#print(orientation(x, y, L, Dx, Dy))
#print(orientation(x+e, y+e, L, Dx, Dy))
#print(orientation(x-e, y-e, L, Dx, Dy))

def build_descriptor(x, y, L, Dx, Dy):
    result  = np.zeros((4,4,4))
    theta   = orientation(x, y, L, Dx, Dy)
    gauss   = linear_scale_space(1, 4)
    sigma   = int(0.4 * L)
    coords  = np.array([[[(x-10) * sigma, (y-10) * sigma] for y in range(20)] for x in range(20)])
    rotate  = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    coords  = np.rint(np.dot(coords, rotate)).astype(int)
    for i in range(4):
        for j in range(4):
            for du in range(5):
                for dv in range(5):
                    u, v    = coords[5 * i + du, 5 * j + dv]
                    su, sv  = sigma * u, sigma * v
                    dx, dy  = gauss[u // 3, v // 3] * np.array([Dx[x + su, y + sv], Dy[x + su, y + sv]])
                    result[i,j] += np.array([dx, dy, np.abs(dx), np.abs(dy)])
    result = result / np.where(result != 0, la.norm(result, axis=2).reshape(4,4,1), 1)
    return result

#x   = image.shape[0] // 2
#y   = image.shape[1] // 2
#L   = 3
#Dx  = first_order_x_image(L, integral)
#Dy  = first_order_y_image(L, integral)
#
#e = 10
#print(build_descriptor(x, y, L, Dx, Dy))
#print(build_descriptor(x+e, y+e, L, Dx, Dy))
#print(build_descriptor(x-e, y-e, L, Dx, Dy))
#print(la.norm(build_descriptor(x, y, L, Dx, Dy) - build_descriptor(x+e, y+e, L, Dx, Dy)))

from time import time
import sys


Ls      = np.array([3,5,7,9,13,17,25,33,49,65])
mats    = []


t = time()
integral = np.cumsum(np.cumsum(image, axis=0), axis=1)
print(time() - t)
sys.stdout.flush()


t = time()
for L in Ls:
    hessian = determinant_of_hessian_image(L, integral)
    Dx      = first_order_x_image(L, integral)
    Dy      = first_order_y_image(L, integral)
    mats.append((hessian, Dx, Dy))
print(time() - t)
sys.stdout.flush()


features    = []
t = time()
for i in range(1, len(Ls) - 1):
    extracted = feature_selection(mats[i][0], mats[i - 1][0], mats[i + 1][0], Ls[i])
    features += extracted
    print(Ls[i], len(extracted))
    sys.stdout.flush()

print(len(features))
print(time() - t)
sys.stdout.flush()


descriptors     = []
t = time()
for feature in features:
    print(feature)
    iL  = np.abs(Ls - feature[2]).argmin
    descriptors.append(build_descriptor(feature[0], feature[1], feature[2], mats[i][1], mats[i][2]))
print(time() - t)
sys.stdout.flush()