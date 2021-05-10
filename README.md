# formidrive_python
Afin de ne pas surcharger le serveur ou sera déposé l'API, on passe par un docker qui contient la partie server, cela évite d'avoir à le compiler (prend plusieurs heures au lieu de quelques secondes sur mon pc)

## docker
je vous ai fournis:
- *le Dockerfile*
- *build.sh* le fichier pour le compiler 
- *run.sh* le fichier pour lancer le docker en local sur le port *8080*
- accès de l'API sur */predict*
le docker lance ensuite le fichier server.py

## scripts pour tester
- client.py passe par sur le site *api.govolum.com* pour envoyer la première photo (appelée face.jpg dans le répertoire courant)
- client2.py passe par le site *api.govolum.com* pour l'envoie des 2 photos avec numéro aléatoire
- localcomplet.py passe par le docker *cadotinfo/opencv* et envoie 2 photos avec numéro aléatoire

## Erreurs
une erreur *Max retries exceeded with url* siginfie que le serveur n'est pas lancé

## Répertoire
- *Images* contient le logo en n/b et couleur avec différent taux de compression (cela à permis de gagner en vitesse)
- *images_etapes* contient les images intermédiares des calculs pur débuguage
- *images_original* contient les images original, permet de retester en local si un problème c'est passé

## Procedure pour tester en local
Vous avez fait des modifications dans server.py.
Il suffit de lancer run.sh, il recompile automatiquement le docker et relance le serveur en local avec vos nouvelles modifications
Bien vérifier par `docker ps` que le serveur est lancer, s'il n'est pas relancé c'est que votre modification à provoqué une erreur dans server.py

## Photos
Les photos doivent prendre la plus grande taille possible sur la photo sans toucher les bords, dans l'idéal.

On peut prendre un carton, ou un frogo, un tas de cartons... en mettant une etiquette avec formidrive dessus (imprimer en couleur ou noir et blanc)

Les photos envoyées seront convertis en noir et blanc  avec une taille de 1000 en largeur(même si mon algo acceptent les photos couleurs et toutes les tailles ) je propose que dans les versions suivantes on retire ces conversions cela nous fera gagner 2 secondes sur les versions suivantes de l'algo)

## retour json
il y a 2 partie, le debug qui donne les temps de chaque partie et les aires trouvées.

## exemple en Curl
pour local
` curl -X POST -F 'id=1234' -F 'photo=@/home/michael/sites/formidrive_python/face.jpg' http://localhost:8080/predict `
pour  l'api
` curl -X POST -F 'id=1234' -F 'photo=@/home/michael/sites/formidrive_python/face.jpg' https://api.govolum.com/predict `

