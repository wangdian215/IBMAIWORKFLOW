#!/usr/bin/env python

import unittest
import re
import numpy as np

# function or class to be tested
def slr_predict(x_query):
    """
    given a simple linear regression make a prediction for x
    """
    
    if isinstance(x_query,float):
        x_query = np.array([x_query])
    elif isinstance(x_query,str):
        if not re.search("\d+",x_query):
            raise Exception("non-numeric string input provided")
        x_query = np.array([float(x_query)])
    elif isinstance(x_query,list):
        x_query = np.array(x_query)
        
    ## generate data for linear regression
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([-1, 0.15, 0.95, 2.1, 2.8])
    
    ## estimate the coeffs using lstsq
    A = np.vstack([x, np.ones(len(x))]).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

    return(coeffs[0] + (coeffs[1] * x_query))
    
class TestSimpleLinearRegressionPredict(unittest.TestCase):

    # example test method
    def test_numeric(self):
        y_pred = slr_predict(0.5)
        self.assertEqual(0.5,y_pred[0])

    def test_str(self):
        y_pred = slr_predict('0.5')
        self.assertEqual(0.5,y_pred[0])

    def test_list(self):
        y_pred = slr_predict([0.5,0.1])
        self.assertEqual(0.5,y_pred[0])
    
    def test_array(self):
        y_pred = slr_predict(np.array([0.5,0.1]))
        self.assertEqual(0.5,y_pred[0])

        
### Run the tests
if __name__ == '__main__':
    unittest.main()        

        