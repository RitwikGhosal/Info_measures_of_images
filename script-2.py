import cv2
import pandas
import numpy as np
from itertools import product

img = cv2.imread("G:/Image_2022/fashion/images_BW/1.jpg",cv2.IMREAD_UNCHANGED)
print(img)
#------------------------------creating the 16 possible matrices-------------------------------------------
a = list(product(range(0,256,255), repeat = 4))
re_matrices = []
matrices = np.matrix(a)
for i in matrices:
    re_matrices.append(np.reshape(i,(2,2)))

image_matrices = []
for i in range(0,28,2):
    for j in range(0,28,2):
        image_matrices.append(img[i:i+2,j:j+2])
        #print(image_matrices)
#print(re_matrices[0])

same = []
for i in range(0,16):
    for j in range(0,196):

        comparison = np.matrix(image_matrices[j]) == np.matrix(re_matrices[i])
        equal = (np.array(comparison)).all()
        
        if equal == True:
            same.append('similar_{}'.format(i+1))

prob = []
for i in range(0,16):       
    #print(same.count('similar_{}'.format(i+1)))
    probab = same.count('similar_{}'.format(i+1)) / 196
    print('probability of {}th matrix is {}'.format(i+1,probab))
    prob.append(same.count('similar_{}'.format(i+1)))    