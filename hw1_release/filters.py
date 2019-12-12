"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np
import math


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    kernel = np.flip(np.flip(kernel, 0), 1)
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            if i < Hk // 2 or i >= (Hi - Hk// 2)  or j < Wk // 2 or j >= (Wi - Wk// 2) :
                continue
            out[i, j] = np.sum(np.multiply(kernel, image[i - Hk // 2:i + 1 + Hk // 2, j - Wk // 2: j + 1 + Wk // 2]))

            ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.concatenate((np.zeros((H, pad_width)), image, np.zeros((H, pad_width))), axis=1)
    out = np.concatenate((np.zeros((pad_height, W+pad_width*2)), out, np.zeros((pad_height, W+pad_width*2))))
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    Hi, Wi = f.shape
    Hk, Wk = g.shape
#     print(g.shape)
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            if i < Hk // 2 or i >= (Hi - Hk// 2)  or j < Wk // 2 or j >= (Wi - Wk// 2) :
                continue
#             print(i)
            out[i, j] = np.sum(np.multiply(g, f[i - Hk // 2:i + Hk // 2, j - Wk // 2: j + 1 + Wk // 2]))

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = g - np.mean(g)
    out = cross_correlation(f,g)
    pass

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = (g - np.mean(g)) / np.std(g)
    Hi, Wi = f.shape
    Hk, Wk = g.shape
#     print(g.shape)
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            if i < Hk // 2 or i >= (Hi - Hk// 2)  or j < Wk // 2 or j >= (Wi - Wk// 2) :
                continue
#             print(i)
            patch = f[i - Hk // 2:i + Hk // 2, j - Wk // 2: j + 1 + Wk // 2]
            patch = (patch - np.mean(patch)) / np.std(patch)
            out[i, j] = np.sum(np.multiply(g, patch))
    ### END YOUR CODE

    return out
