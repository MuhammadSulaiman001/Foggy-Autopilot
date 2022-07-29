import tensorflow
import scipy
import numpy
import cv2
import matplotlib
import sklearn
import h5py
import unittest

class MiscTests(unittest.TestCase):
    def test1(self):
        print(cv2.__version__)
        print(numpy.version.version)
        print(sklearn.__version__)
        print(tensorflow.__version__)
        print(matplotlib.__version__)
        print(scipy.__version__)
        print(h5py.__version__)

if __name__ == '__main__':
    unittest.main()
