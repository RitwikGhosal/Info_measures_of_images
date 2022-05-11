import cv2
import numpy as np
from itertools import product
import matplotlib.pyplot as mt


img = cv2.imread("G:/Image_2022/fashion/images_aq/1.jpg",cv2.IMREAD_GRAYSCALE)
#print(img)
#------------------------------creating the 16 possible matrices-------------------------------------------
a = list(product(range(0,256,255), repeat = 4))
re_matrices = []
matrices = np.matrix(a)
for i in matrices:
    re_matrices.append(np.reshape(i,(2,2)))
    
#print(re_matrices)

#-----------------------------------------------------------------------------------------------------------
#-----------------------------dynamic otsu's threshold------------------------------------------------------
bins_num = 256
hist, bin_edges = np.histogram(img, bins=bins_num)
mt.plot(hist)
#if is_normalized:
#    hist = np.divide(hist.ravel(), hist.max())
bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
weight1 = np.cumsum(hist)
weight2 = np.cumsum(hist[::-1])[::-1]
mean1 = np.cumsum(hist * bin_mids) / weight1
mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
index_of_max_val = np.argmax(inter_class_variance)
threshold = bin_mids[:-1][index_of_max_val]
print("Otsu's algorithm implementation thresholding result: ", threshold)

#------------------------------converting the image to binary-------------------------------------------------
ret, thresh1 = cv2.threshold(img, threshold, 255, cv2.THRESH_OTSU) 
#ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 
#------------------------------having the all 2x2 matrices of the binary images--------------------------------
#print(type(thresh1))
#print(thresh1)
image_matrices = []
for i in range(0,28,2):
    for j in range(0,28,2):
        image_matrices.append(thresh1[i:i+2,j:j+2])
        #print(image_matrices)
#print(re_matrices[0])

c_1 = 0
c_2 = 0

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

#print(tuple(prob))
#print(same)
#rint(c_2)
#print(np.array((image_matrices[j])).reshape(-1))
#cv2.imwrite("G:/Image_2022/fashion/new.jpg",thresh1)
#img_1 = cv2.imread("G:/Image_2022/fashion/new.jpg",0)

#print(img_1)
#cv2.imshow('hello',thresh1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(type(ret))
