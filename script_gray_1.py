import cv2
import numpy as np
from itertools import product
import pandas as pd

count = 0
a = list(product(range(0,256,255), repeat = 4))
re_matrices = []
matrices = np.matrix(a)
for i in matrices:
    re_matrices.append(np.reshape(i,(2,2)))

prob_1 = []

for i in range(1,50):
    img = cv2.imread("G:/Image_2022/gray49/images_aq_gray/gray49/im{}.jpg".format(i),cv2.IMREAD_GRAYSCALE)
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
    #print("Otsu's algorithm implementation thresholding result: ", threshold)
    count+=1
    ret, thresh1 = cv2.threshold(img, threshold, 255, cv2.THRESH_OTSU) 
    #cv2.imwrite(r'G:/Image_2022/gray49/imgaes_gray_BW'+ str(count) +'.jpg',thresh1)

    image_matrices = []
    for j in range(0,512,2):
        for k in range(0,512,2):
            image_matrices.append(thresh1[j:j+2,k:k+2])

    same = []
    for l in range(0,16):
        for m in range(0,65536):

            comparison = np.matrix(image_matrices[m]) == np.matrix(re_matrices[l])
            equal = (np.array(comparison)).all()
            
            if equal == True:
                same.append('similar_{}'.format(l+1))

    prob = []
    
    #fields = []
    for n in range(0,16):       
        #print(same.count('similar_{}'.format(i+1)))
        probab = same.count('similar_{}'.format(n+1)) / 65536
        prob.append(probab)
    prob_1.append(prob)
        #print('probability of {}th matrix is {}'.format(n+1,probab))
        #prob.append(probab)

#print(prob_1)
print(len(prob_1))

array = np.array(prob_1)
index_values = list(map(str,np.arange(1,50)))
column_values = list(map(str,np.arange(1,17)))

#print(array)

df = pd.DataFrame(data = array, 
                  index = index_values, 
                  columns = column_values)

df.to_csv('Images_probab_gray49_revised.csv', index=False,header=False)