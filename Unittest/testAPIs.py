#!/usr/bin/env python
"""

import sys
import os
import unittest
import requests
import re
from ast import literal_eval
import numpy as np

port = 8080

try:
    requests.post('http://0.0.0.0:{}/predict'.format(port))
    server_available = True
except:
    server_available = False
    
## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    test the essential functionality
    """
    
    @unittest.skipUnless(server_available,"local server is not running")
    def test_predict_empty(self):
           
        r = requests.post('http://0.0.0.0:{}/predict'.format(port))
        self.assertEqual(re.sub('\n|"','',r.text),"[]")

        r = requests.post('http://0.0.0.0:{}/predict'.format(port),json={"key":"value"})     
        self.assertEqual(re.sub('\n|"','',r.text),"[]")
    
    @unittest.skipUnless(server_available,"localhost server is not running")
    def test_predict(self):
        
	query_data = ("germany", '2018', '8', '5')
        query_type = 'tuple'
        request_json = {'query':query_data,'type':query_type}

        r = requests.post('http://0.0.0.0:{}/predict'.format(port),json=request_json)
        response = literal_eval(r.text)
        self.assertEqual(response['y_pred'],[1])

    @unittest.skipUnless(server_available,"local server is not running")
    def test_train(self):
        
        request_json = {'mode':'test'}
        r = requests.post('http://0.0.0.0:{}/train'.format(port),json=request_json)
        train_complete = re.sub("\W+","",r.text)
        self.assertEqual(train_complete,'true')

### Run the tests
if __name__ == '__main__':
    unittest.main()
