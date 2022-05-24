import cv2
import numpy as np
from itertools import product
import pandas as pd

count = 0
a = list(product(range(0,256,255), repeat = 16))
re_matrices = []
matrices = np.matrix(a)
for i in matrices:
    re_matrices.append(np.reshape(i,(4,4)))

prob_1 = []

for i in range(1000,1011):
    img = cv2.imread("G:/Image_2022/fashion/images_aq/{}.jpg".format(i),cv2.IMREAD_GRAYSCALE)
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
    #cv2.imwrite(r'G:/Image_2022/fashion/images_BW_new/'+ str(count) +'.jpg',thresh1)

    
    image_matrices = []
    for j in range(0,28,4):
        for k in range(0,28,4):
            image_matrices.append(thresh1[j:j+4,k:k+4])

    same = []
    for l in range(0,65536):
        for m in range(0,49):

            comparison = np.matrix(image_matrices[m]) == np.matrix(re_matrices[l])
            equal = (np.array(comparison)).all()
            
            if equal == True:
                same.append('similar_{}'.format(l+1))

    prob = []
    
    #fields = []
    for n in range(0,65536):       
        #print(same.count('similar_{}'.format(i+1)))
        probab = same.count('similar_{}'.format(n+1)) / 49
        if probab != 0:
            print('probability of {}th matrix is {}'.format(n+1,probab))
        #prob.append(probab)
    #prob_1.append(prob)
        #print('probability of {}th matrix is {}'.format(n+1,probab))
        #prob.append(probab)

#print(prob_1)
#print(len(prob_1))
'''
array = np.array(prob_1)
index_values = list(map(str,np.arange(1,2)))
column_values = list(map(str,np.arange(1,65537)))

#print(array)

df = pd.DataFrame(data = array, 
                  index = index_values, 
                  columns = column_values)

print(df)
df.to_csv(r'Images_probab_four_four.txt', index=False,header=False,sep=',', mode='a')
'''