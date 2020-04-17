''''
A script for sending a JSON in the POST request
'''

import requests
import json

path = "http://127.0.0.1:80/status"


folder = {"base": "C:/Users/azfar/Downloads/samuel/test2"}

r = requests.post(path, data = folder)
print(r.text)