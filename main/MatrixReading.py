# MatrixReading.py
#
# File for reading a sparse SPAM dataset into julia.
# translated from MatrixReading.py

import numpy as np
from scipy.sparse import csr_matrix


def read_matrix(filename):
    """ (sp_matrix, tokenlist, category) = ReadMatrix(filename)

    Reads the file stored at `filename`, which is of the format of
    MATRIX.TEST, and returns a 3-tuple. The first part is 'sp_matrix',
    an m-by-n sparse matrix, where m is the number of training/testing
    examples and n is the dimension, and each row of sp_matrix consists
    of counts of word appearances. (So sp_matrix[i, j] is the number of
    times word j appears in document i.)

    tokenlist is a list of the words, where tokenlist[1] is the first
    word in the dictionary and tokenlist[end] is the last.

    ategory is a {0, 1}-valued vector of positive and negative
    examples. Before using in SVM code, you should transform categories
    to have signs +/-1.
    """
    with open(filename) as fstream:
        # Read header line, discard
        next(fstream)
        # Read rows and columns, turn into integers
        num_rows, num_cols = tuple(map(int, next(fstream).split()))
        # Read the list of tokens - just a long string!
        tokenlist = next(fstream).split()

        # Now to read the matrix into the matrix. Each row represents a
        # document (mail), each column represents a distinct token. As the
        # data isn't actually that big, we just use a full matrix to save
        # time.
        full_mat = np.zeros((num_rows, num_cols))
        categories = np.zeros(num_rows, dtype='int')
        for ii in range(num_rows):
            row = np.fromiter(map(int, next(fstream).split()), dtype=int)
            categories[ii] = int(row[0])
            jj = 1
            offset = 0
            while row[jj] != -1:
                offset += row[jj]
                count = row[jj + 1]
                full_mat[ii, offset] = count
                jj += 2;

    return (csr_matrix(full_mat), tokenlist, categories)