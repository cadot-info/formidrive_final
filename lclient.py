import requests
import os
import io
from PIL import Image
import base64
import logging

#url = 'http://formidrive.cadot.info:8080/predict'
url = 'http://localhost:8080/predict'

with open('tf.jpg', 'rb') as f:
    photo = base64.b64encode(f.read())

post = {'photo': photo, 'id': '12345ab'}

r = requests.post(url, post)

# convert server response into JSON format.
print(r)
print(r.json())
