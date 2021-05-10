# Utilisation de l'api.

Je l'ai mis sur mon serveur de site web, il est pas assez costaud pour ce type d'applis.
Donc on est au pire.

## amélioration future
Une découpe est trop longue (elle prend plus de 90% du temps.
Je vais travaillé dessus.

## Url de l'ap
En attendant, voici l'adresse https://formidrive.cadot.info/persist
l'api fonctionne sur le port 80 
Exemple d'envoie (une image par une avec un id unique pour les deux)
Cela permettra à l'utilisateur d'envoyer une photo, il prend la deuxième, pendant ce temps il a calculé la première ;-)
On divise ainsi le temps de calcul par 2.

## Photos
Les photos doivent prendre la plus grande taille possible sur la photo sans toucher les bords, dans l'idéal.

On peut prendre un carton, ou un frogo, un tas de cartons... en mettant une etiquette avec formidrive dessus (imprimer en couleur ou noir et blanc)

Les photos envoyées seront convertis en noir et blanc  avec une taille de 1000 en largeur(même si mon algo acceptent les photos couleurs et toutes les tailles ) je propose que dans les versions suivantes on retire ces conversions cela nous fera gagner 2 secondes sur les versions suivantes de l'algo)

## Sécurité
Pour la sécurité, l'api vérifiera l'origine de la demande (application_id) 

## Code exemple
Je vous donne un code python comme exemple pour envoyer une photo et mettre un id


import requests
import os
import io
from PIL import Image
import base64

url = 'https://formidrive.cadot.info:80/predict'

with open('photo.jpg', 'rb') as f:
    photo = base64.b64encode(f.read())

post = {'photo': photo, 'id': '12345ab'}

r = requests.post(url, post)


print(r.json())
