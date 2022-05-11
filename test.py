from itertools import product
import numpy as np
from collections import Counter

#a = list(product(range(0,256,255), repeat = 4))

#matrices = np.matrix(a)
#for i in matrices:
#    re_matrices = np.reshape(i,(2,2))
#    print(re_matrices)

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

matr = []
for i in range(1,17):
    matr.append('matrix_{}'.format(i))

print(matr)

pro = [31200.01531,647.3520434,529.6530662,1009.571429,435.7857207,1704.556124,60.96938934,717.3724509,462.4898015,55.83163409,1828.653062,605.7244939,1212.086736,518.6479643,542.3469435,18439.35714]

#print(pro)
#print(len(pro))

def listOfTuples(l1, l2):
    return list(map(lambda x, y:(x,y), l1, l2))


print(listOfTuples(matr, pro))
if __name__ == '__main__':
    #freq = dict(Counter(string))
    freq = listOfTuples(matr, pro)
    node = make_tree(freq)
    encoding = huffman_code_tree(node)
    for i in encoding:
        print(f'{i} : {encoding[i]}')

#print(type(freq[0]))


