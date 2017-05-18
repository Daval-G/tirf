import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import os
import numpy as np
import numpy.linalg as la
import sys
import math
import classifier as cl
from math import sqrt

def apply_convol(image, filter):
    result = np.copy(image)
    
    dx = filter.shape[0]
    dy = filter.shape[1]
    
    for x in range(0, image.shape[0] - dx):
        for y in range(0, image.shape[1] - dy):
            for c in range(3):
                result[x + dy // 2, y + dy // 2, c] = min(255, max(0, np.sum(np.multiply(filter, image[x: x + dx, y: y + dy, c]))))
    return result

def flood_fill(x, y, grad, image):
    if (np.sum(image[x, y]) == 0):
        return image

    result = np.copy(image)
    
    next = [[[x, y],[x, y],0]]
    
    while (next != []):
        nx, ny = next[0][0]
        px, py = next[0][1]
        d = next[0][2]
        
        if (nx < 0 or ny < 0 or nx >= image.shape[0] or ny >= image.shape[1]
                or abs(nx - x) + abs(ny - y) != d
                or np.sum(result[nx, ny]) == 0):
            next = next[1:]
            continue
        
        next = next[1:]        
        diff= np.sqrt(np.sum(np.abs(image[nx, ny] - image[x, y]) ** 2))
        
        if (diff < grad):
            result[nx, ny] = 0

            next.append([[nx + 1, ny], [nx, ny], d+1])
            next.append([[nx - 1, ny], [nx, ny], d+1])
            next.append([[nx, ny + 1], [nx, ny], d+1])
            next.append([[nx, ny - 1], [nx, ny], d+1])

    return result

def home_maid_distance(color):
    maid_colors = np.array([[70,50,90], [80,80,80], [250,250,250], [50,50,50], [180, 190, 75], [240, 240, 200]])
    
    #maid_colors = maid_colors / np.linalg.norm(maid_colors, axis=1)[:, np.newaxis]
    #color       = color / np.linalg.norm(color)
    return np.min(np.sqrt(np.sum(np.abs(color - maid_colors) ** 2, axis=1)))

#brightness = np.mean(np.sqrt(np.dot(im[:,:] ** 2, np.array([0.241,0.691,0.068]))))
#ref_mean = np.mean(np.mean(im, axis=1), axis=0)
#ref_deviation = np.mean(np.mean(np.abs(im - ref_mean), axis=1), axis=0)
#ref_mean = ref_mean
#ref_deviation = ref_deviation / np.linalg.norm(ref_deviation)

#from ml_toolkit import *
#from classifier import *
#
#shape = 10
#
#classifier  = Gen_Gauss() #NeuralNetwork([shape * shape * 3, 100, 100, 1], activation=Sigmoid)
#data        = []
#labels      = []
#
#for file in ["reference06.jpg", "reference07.jpg", "reference08.jpg"]:
#    im = imread("test/" + file)
#    for x in range(im.shape[0] - shape):
#        for y in range(im.shape[1] - shape):
#            data.append(np.array([im[x: x + shape, y: y + shape].reshape((shape * shape * 3,))]))
#            labels.append(1)
#print("MAID DATA")
#sys.stdout.flush()
#
#for file in ["reference03.jpg", "reference04.jpg"]:
#    im = imread("test/" + file)
#    for x in range((im.shape[0] - 1) // shape):
#        for y in range((im.shape[1] - 1) // shape):
#            data.append(np.array([im[x * shape: (x+1) * shape, y * shape: (y+1) * shape].reshape((shape * shape * 3,))]))
#            labels.append(0)
#print("UNMAID DATA")
#sys.stdout.flush()
#
#data = np.array(data)
#data = data.reshape((data.shape[0], data.shape[2]))
#np.random.shuffle(data)
#print(data.shape)
#sys.stdout.flush()

#import matplotlib.image as mpimg
#
#img = mpimg.imread('test/reference02.jpg')
#gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
#filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
#gray = apply_convol(gray, filter)
#plt.imshow(gray, cmap = plt.get_cmap('gray'))
#plt.show()
#gray = apply_convol(gray, filter)
#plt.imshow(gray, cmap = plt.get_cmap('gray'))
#plt.show()

#import cPickle as pk

#f = open("classifier.pkl", "r")
#classifier = pk.load(f)
#f.close()


#for i in range(1):
#    print(classifier.train(data, np.array(labels), epochs=1))
#    print(i)
#    sys.stdout.flush()
#    f = open("classifier.pkl", "w")
#    pk.dump(classifier, f)
#    f.close()
#classifier.train(data.T, labels)
#print("TRAINED")
#sys.stdout.flush()

#im = imread("test/facile/20160524_163619.jpg")
#for file in os.listdir("test/facile"):
#    im = imread("test/facile/" + file)
#    
#    copy = np.copy(im)
#    
#    for x in range((im.shape[0] - 1) // shape):
#        print(x)
#        sys.stdout.flush()
#        for y in range((im.shape[1] - 1) // shape):
#            label = classifier.process(im[x * shape: (x+1) * shape, y * shape: (y+1) * shape].reshape((1, shape * shape * 3)))
#            if (label[0] != 1):
#                copy[x * shape: (x+1) * shape, y * shape: (y+1) * shape] = 0
#    
#    plt.imshow(copy)
#    plt.show()

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

#import PIL
#from PIL import Image

#im = Image.open('test/reference11.jpg')
#tau = float(im.size[0]) / im.size[1]
#im = im.resize((int(1000 * tau), 1000), PIL.Image.ANTIALIAS)
#im.save('test/reference-2.jpg')

#im = imread("test/reference-2.jpg")
#mins = 100000 * 255 * np.ones((im.shape[0], im.shape[1]))

#filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
#im = apply_convol(im, filter)
#plt.imshow(im)
#plt.show()
#
#for i in range(im.shape[1]):
#    mins = dist_fill(0, i, mins, im)
#    mins = dist_fill(im.shape[0] - 1, i, mins, im)
#plt.imshow(mins, cmap = plt.get_cmap('gray'))
#plt.show()
#
#for i in range(im.shape[0]):
#    mins = dist_fill(i, 0, mins, im)
#    mins = dist_fill(i, im.shape[1] - 1, mins, im)
#plt.imshow(mins, cmap = plt.get_cmap('gray'))
#plt.show()
#
#maxi = np.max(mins)
#limi = maxi // 4
#for i in range(mins.shape[0]):
#    for j in range(mins.shape[1]):
#        mins[i, j] = (mins[i, j] // limi)
#plt.imshow(mins, cmap = plt.get_cmap('gray'))
#plt.show()

#for x in range(im.shape[0]):
#    for y in range(im.shape[1]):
#        im[x, y] = im[x, y] * mins[x, y] // maxi
#plt.imshow(im)
#plt.show()
#
#for x in range(im.shape[0]):
#    for y in range(im.shape[1]):
#        val = home_maid_distance(im[x, y])
#        if (val > 20):
#            im[x, y] = 0
#plt.imshow(im)
#plt.show()

#imsave("test/facile/test_" + file, im)



#------------------------#
#-----SURF ALGORITHM-----#
#------------------------#
image = imread('test/reference02.jpg')

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
# PIXEL SCALE
def first_order_y_box_filter(x, y, l, i):
    return (i[x + l, y + l] + i[x - l - 1, y] - i[x + l, y] - i[x - l - 1, y + l]) \
         - (i[x + l, y - 1] + i[x - l - 1, y - l - 1] - i[x + l, y - l - 1] - i[x - l - 1, y - 1])
# Note that l = round(0.8L)

# IMAGE SCALE
def first_order_x_image(L, integral):
    return first_order_y_image(L, integral.T).T

def first_order_y_image(L, integral):
    l = int(0.8 * L)
    result = np.zeros(integral.shape)
    for x in range(l + 1, integral.shape[0] - l):
        for y in range(l + 1, integral.shape[1] - l):
            result[x, y] = first_order_y_box_filter(x, y, l, integral)
    return result

#L = 5
#plt.imshow(first_order_x_image(L, integral), cmap = plt.get_cmap('gray'))
#plt.show()
#plt.imshow(first_order_y_image(L, integral), cmap = plt.get_cmap('gray'))
#plt.show()

# SECOND ORDER BOX FILTER
# PIXEL SCALE    
def second_order_yy_box_filter(x, y, L, i):
    L_2 = L // 2
    L3_2 = 3 * L_2
    return (i[x + L, y + L3_2] + i[x - L - 1, y - L3_2 - 1])    \
         + (i[x - L - 1, y + L_2] + i[x + L, y - L_2 - 1]) * 3  \
         - (i[x + L, y + L_2] + i[x - L - 1, y - L_2 - 1]) * 3  \
         - (i[x - L - 1, y + L3_2] + i[x + L, y - L3_2 - 1])
    
def second_order_xy_box_filter(x, y, L, i):
    return (i[x + L, y + L] + i[x, y])                  \
         + (i[x - 1, y - 1] + i[x - L - 1, y - L - 1])  \
         + (i[x - 1, y] + i[x - L - 1, y + L])          \
         + (i[x, y - 1] + i[x + L, y - L - 1])          \
         - (i[x, y + L] + i[x + L, y])                  \
         - (i[x - 1, y - L - 1] + i[x - L - 1, y - 1])  \
         - (i[x - 1, y + L] + i[x - L - 1, y])          \
         - (i[x + L, y - 1] + i[x, y - L - 1])

# IMAGE SCALE
def second_order_xy_image(L, integral):
    result = np.zeros(integral.shape)
    for x in range(L + 1, integral.shape[0] - L):
        for y in range(L + 1, integral.shape[1] - L):
            result[x, y] = second_order_xy_box_filter(x, y, L, integral)
    return result

def second_order_xx_image(L, integral):
    return second_order_yy_image(L, integral.T).T

def second_order_yy_image(L, integral):
    result = np.zeros(integral.shape)
    for x in range(3 * L // 2 + 1, integral.shape[0] - 3 * L // 2 - 1):
        for y in range(3 * L // 2 + 1, integral.shape[1] - 3 * L // 2 - 1):
            result[x, y] = second_order_yy_box_filter(x, y, L, integral)
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
        or (hessian[x, y] >= hessian[x + 1, y - 1] and hessian[x, y] >= hessian[x - 1, y + 1]) \

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
    thres = np.max(hessian) // 2
    features = []
    for x in range(hessian.shape[0]):
        for y in range(hessian.shape[1]):
            if (hessian[x, y]) > thres:
                if non_maximum_suppression(x, y, hessian):
                    pt = location_refinement(x, y, L, hessian, prev_hessian, next_hessian)
                    if not (pt is None):
                        features.append((int(pt[0]), int(pt[1]),int(pt[2])))                    
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
    print(la.norm(result, axis=2))
    print(result)
    result = result / la.norm(result, axis=2)
    print(result)
    return result

x   = image.shape[0] // 2
y   = image.shape[1] // 2
L   = 3
Dx  = first_order_x_image(L, integral)
Dy  = first_order_y_image(L, integral)

e = 10
#print(build_descriptor(x, y, L, Dx, Dy))
#print(build_descriptor(x+e, y+e, L, Dx, Dy))
#print(build_descriptor(x-e, y-e, L, Dx, Dy))
print(la.norm(build_descriptor(x, y, L, Dx, Dy), axis=2))