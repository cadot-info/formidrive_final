import requests
import os
import io
from PIL import Image
import base64
import logging
import random2
import sys
from random2 import *

logging.basicConfig(level=logging.DEBUG)

url = 'http://formidrive.cadot.info:8080/predict'

with open('face.jpg', 'rb') as f:
    photo = base64.b64encode(f.read())
nbr=randint(0,99999)
print(nbr)
post = {'photo': photo, 'id': nbr}

r = requests.post(url, post)
print(r.json())
with open('cote.jpg', 'rb') as f:
    photo = base64.b64encode(f.read())

post = {'photo': photo, 'id': nbr}
r = requests.post(url, post)
print(r.json())

