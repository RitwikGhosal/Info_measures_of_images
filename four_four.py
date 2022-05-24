import cv2
import numpy as np
from itertools import product

a = list(product(range(0,256,255), repeat = 16))
re_matrices = []
matrices = np.matrix(a)
for i in matrices:
    re_matrices.append(np.reshape(i,(4,4)))
    #print(np.reshape(i,(4,4)))

#print(len(re_matrices))

#img = cv2.imread("G:/Image_2022/fashion/images_aq/2.jpg",cv2.IMREAD_GRAYSCALE)
img = cv2.imread("G:/Image_2022/sample/im17.jpg",cv2.IMREAD_GRAYSCALE)

#image_matrices = []
#for j in range(0,56,4):
  #  for k in range(0,56,4):
  #      image_matrices.append(img[j:j+4,k:k+4])

#print(len(image_matrices))

bins_num = 256
hist, bin_edges = np.histogram(img, bins=bins_num)
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
#print(re_matrices[0])
image_matrices = []
for j in range(0,56,4):
    for k in range(0,56,4):
        image_matrices.append(thresh1[j:j+4,k:k+4])

same = []
for l in range(0,65536):
    for m in range(0,196):

        comparison = np.matrix(image_matrices[m]) == np.matrix(re_matrices[l])
        equal = (np.array(comparison)).all()
        
        if equal == True:
            same.append('similar_{}'.format(l+1))

prob = []
count_lst = []
for n in range(0,65536):       
    #same.count('similar_{}'.format(n+1))
    probab = same.count('similar_{}'.format(n+1)) / 196
    if probab != 0:
        print('probability of {}th matrix is {}'.format(n+1,probab))
    #prob.append(same.count('similar_{}'.format(n+1)))


