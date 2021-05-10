import requests
import os
import io
from PIL import Image
import base64
import logging

logging.basicConfig(level=logging.DEBUG)

url = 'http://formidrive.cadot.info:8080/predict'
#url = 'http://localhost:8080/predict'

with open('cote.jpg', 'rb') as f:
    photo = base64.b64encode(f.read())


#payload = {'id': '123', 'type': 'jpg','box': [0, 0, 100, 100], 'image': im_b64}
post = {'photo': photo, 'id': '12345ab'}

r = requests.post(url, post)

# convert server response into JSON format.
print(r.json())
