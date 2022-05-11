import pandas
import numpy as np
import cv2
from itertools import product

data = pandas.read_csv("G:/Image_2022/fashion/fashion-mnist_train.csv")

print(data.shape[0])
#------------------------------------------------generating all possible 2x2 matrices--------------------------------------------------
a = list(product(range(0,256,255), repeat = 4))
re_matrices = []
matrices = np.matrix(a)
for i in matrices:
    re_matrices.append(np.reshape(i,(2,2)))

#------------------------------------------------converting each image to binary using otsu---------------------------------------------
def convert2image(row):
    image = np.array(row[1:785])
    image = image.reshape(28, 28)
    return image.astype(np.uint8)


bins_num = 256
count = 0 
for i in range(1, data.shape[0]): 
    face = data.iloc[i] 
    img = convert2image(face)  
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist, bin_edges = np.histogram(img, bins=bins_num)
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
    #cv2.imwrite(r'G:/Image_2022/fashion/images_aq/'+ str(count) +'.jpg',img)
    ret, thresh1 = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_OTSU) 
    cv2.imwrite(r'G:/Image_2022/fashion/images_BW_new/'+ str(count) +'.jpg',thresh1)


#having all the 2x2 images of the binary image--------------------------------------------------------------------------------------------------

#image_matrices = []
#for j in range(0,28,2):
    #for k in range(0,28,2):
        #image_matrices.append(thresh1[j:j+2,k:k+2])

# finding tyhe probability ---------------------------------------------------------------------------------------------------------------------

#same = []
#for l in range(0,16):
#    for m in range(0,196):

#        comparison = np.matrix(image_matrices[l]) == np.matrix(re_matrices[m])
#        equal = (np.array(comparison)).all()
        
#        if equal == True:
#            same.append('similar_{}'.format(i+1))

#for n in range(0,16):       
    #print(same.count('similar_{}'.format(i+1)))
#    probab = same.count('similar_{}'.format(n+1)) / 196
#    print('probability of {}th matrix is {}'.format(n+1,probab))