import cv2
import numpy as np
from itertools import product
import pandas as pd
import time
import numpy_indexed as npi
from collections import Counter

start = time.time()

#count = 0
'''
a = list(product(range(0,256,255), repeat = 16))
re_matrices = []
matrices = np.matrix(a)
for i in matrices:
    re_matrices.append(np.reshape(i,(4,4)))

final_matrices = [] 
prob_1 = []
'''
final_matrices = []
def otsu_method(image):
    #image = cv2.imread("G:/Image_2022/gray49/images_aq_gray/gray49/im{}.jpg".format(i),cv2.IMREAD_GRAYSCALE)
    bins_num = 256
    hist, bin_edges = np.histogram(image, bins=bins_num)
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
    #count+=1
    ret, thresh1 = cv2.threshold(image, threshold, 255, cv2.THRESH_OTSU)
    return thresh1

def compare(i1, i2):
    m1, c1 = i1[0], i1[1]
    m2, c2 = i2[0], i2[1]
    m = np.vstack((m1, m2))
    
    u = np.unique(m, axis=0, return_counts=True)
    
    arr, c = [], []
    for i in range(len(u[0])):
        
        element = u[0][i]
        index1 = np.flatnonzero(npi.contains([element], m1))
        index2 = np.flatnonzero(npi.contains([element], m2))
        arr.append(element) 
        if u[1][i] == 2:
            c.append(c1[index1[0]]+c2[index2[0]])
        else:
            if len(index2):
                c.append(c2[index2[0]])
            else:
                c.append(c1[index1[0]])
        
        
    return(np.array(arr), np.array(c))
   
def foo(matrix):
    flatten = []
    for i in matrix:
        flatten.append(i.ravel())
        
    #print(flatten)
    flatten = pd.DataFrame(flatten)
    u = np.unique(flatten, axis=0, return_counts=True)
    #print(u)
    return u

# --------------------first image-----------------------------------------    
img_initial = otsu_method(cv2.imread("G:/Image_2022/gray49/images_aq_gray/gray49/im1.jpg",cv2.IMREAD_GRAYSCALE))
for j in range(0,512,8):
    for k in range(0,512,8):
            final_matrices.append((img_initial[j:j+8,k:k+8])) 

past = foo(final_matrices)
#print(final_matrices)

n = len(past[0])
name = [str(i) for i in range(n)]

#counts = []
for i in range(2,50):
    mat = []
    img1 = otsu_method(cv2.imread("G:/Image_2022/gray49/images_aq_gray/gray49/im{}.jpg".format(i),cv2.IMREAD_GRAYSCALE))
    #count = 0
    for j in range(0,512,8):
        for k in range(0,512,8):            
            mat.append(img1[j:j+8,k:k+8])
    #print(np.array(mat).shape)
    unique = foo(mat)
    #print(mat)
    past = compare(past, unique)


#print(past)

class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return self.left, self.right


def huffman_code_tree(node, binString=''):
    '''
    Function to find Huffman Code
    '''
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, binString + '0'))
    d.update(huffman_code_tree(r, binString + '1'))
    return d


def make_tree(nodes):
    '''
    Function to make tree
    :param nodes: Nodes
    :return: Root of the tree
    '''
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return nodes[0][0]

#matr = []
#for i in range(1,17):
#    matr.append('matrix_{}'.format(i))

#print(matr)

pro = list(past[1])

#print(pro)
#print(len(pro))

def listOfTuples(l1, l2):
    return list(map(lambda x, y:(x,y), l1, l2))


#print(listOfTuples(name, pro))

if __name__ == '__main__':
    #freq = dict(Counter(string))
    freq = listOfTuples(name, pro)
    node = make_tree(freq)
    encoding = huffman_code_tree(node)#node
    out_name = []
    out_encoding = []
    for i in encoding:
        #print(f'{i} : {encoding[i]}')
        out_name.append(i)
        out_encoding.append(encoding[i])

    result = pd.DataFrame(np.array([out_name, out_encoding]).T)
    result.to_csv('G:/Image_2022/huffman_8_8_49.csv', header=False, index=False)

end = time.time()
print(end-start)   