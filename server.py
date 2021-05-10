from __future__ import print_function
from waitress import serve
import cv2 as cv2
import os
import base64
import io
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from imutils import contours, perspective
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
import cv2 as cv
import argparse
import time
import requests
import sys
from os import path
import json
import uuid
from flask_cors import CORS

largr = 1000  # taille de l'image pour redimensiponnement
largrmini = 500  # taille pour accélerer, utiliser dans la supperssion du fond
# en cm
# reel_etiquette_x = 19.7
# reel_etiquette_y = 7.4
reel_etiquette_x = 27.8
reel_etiquette_y = 10.5


def resize(image, width=largr):
    height = int(image.shape[0] * width/image.shape[1])
    dim = (width, height)
    return cv.resize(image, dim, interpolation=cv.INTER_AREA)


def remove_background(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(morphed,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]

    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    mask = cv2.drawContours(threshed, cnt, 0, (0, 255, 0), 0)
    masked_data = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(cnt)
    dst = masked_data[y: y + h, x: x + w]

    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(dst_gray, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(dst)

    rgba = [r, g, b, alpha]
    dst = cv2.merge(rgba, 4)

    return dst


app = Flask(__name__)
cors = CORS(app)


@app.route("/", methods=['GET'])
def hello():
    return "API Formidrive!"


@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('photo')
    # return files[0].filename
    destination = 'images_etapes/'
    repdest = ''
    largeur = 0
    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                                              RECUPÉRATIONDE L'IMAGE  #
    # -------------------------------------------------------------------------------------------------------------------- #
    start_time = time.time()    
    # try:
    post = request.form.to_dict(flat=False)
    # recuperation image photo
    # photo_base64 = post['photo'][0]
    # except:
    # return jsonify({'Erreur': 'Image non reçu'})

    try:
        # photo_binary = base64.b64decode(photo_base64)
        # imwrite('base.jpg', request.files.getlist('photo')[0].read)
        request.files.getlist('photo')[0].save(
            os.path.join('', destination+'base.jpg'))
        # photo_binary = cv.img
    except:
        return jsonify({'Erreur': 'Votre image est indécodable'})

    try:
        # photo_base = np.frombuffer(photo_binary, dtype=np.uint8)
        # photo = cv2.imdecode(photo_base, flags=cv2.IMREAD_GRAYSCALE)
        # photo_base = cv.imread(destination+'base.jpg', cv.IMREAD_COLOR)
        photo = cv.imread(destination+'base.jpg', cv.IMREAD_GRAYSCALE)
        debug = {"1-photo recuperee, decodee et convertie": round(
            time.time() - start_time, 2)}
    except:
        return jsonify({'Erreur': 'Votre image est inconvertissable en nuaunce de gris'})

    start_time = time.time()

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                                           CHARGEMENT DE L'IMAGE LOGO #
    # -------------------------------------------------------------------------------------------------------------------- #
    try:
        logo = cv.imread('logo.jpg', cv.IMREAD_GRAYSCALE)
    except:
        return jsonify({'Erreur': 'Votre image est impossible à ouvrir'})

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                   RECUPERATION ID    #
    # -------------------------------------------------------------------------------------------------------------------- #

    try:
        id = post['id'][0]

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                         CONTROLE DE LA PRÉSENCE D'UNE PREMIERE IMAGE #
    # -------------------------------------------------------------------------------------------------------------------- #
        if path.exists(destination+id+'-resultats.json'):
            # on est sur la deuxieme on modifie dest
            dest = destination + id + '-2'
            repdest = 'images_original/' + id + '-2'
        else:
            repdest = 'images_original/' + id + '-1'
            dest = destination + id + '-1'
        # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                                        REDIMENSIONNEMENT DES IMAGES  #
    # -------------------------------------------------------------------------------------------------------------------- #
    except:
        return jsonify({'Erreur': 'Votre image est indétectable'})

    try:
        logo_small = resize(logo)
        photo_small = resize(photo)
        debug.update({"2-redimensionnement des images":
                      round(time.time() - start_time, 2)})
    except:
        return jsonify({'Erreur': 'Votre image est non redimensionnable'})

    start_time = time.time()
    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                        S'IL Y A UN PROBLEME SUR LES IMAGES ON ARRÊTE #
    # -------------------------------------------------------------------------------------------------------------------- #

    if logo_small is None or photo_small is None:
        jsonify({'Erreur': 'Une de vos images est illisible'})
        exit(0)

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                           HOMOGRAPHY #
    # -------------------------------------------------------------------------------------------------------------------- #

    try:
        minHessian = 10  # plus c'est petit plus la taille de l'étiquette est bonne
        detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
        keypoints_obj, descriptors_obj = detector.detectAndCompute(
            logo_small, None)
        keypoints_scene, descriptors_scene = detector.detectAndCompute(
            photo_small, None)
        debug.update({"3-pas2": round(time.time() - start_time, 2)})
        start_time = time.time()
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
        ratio_thresh = 0.75
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        debug.update({"4-SURF": round(time.time() - start_time, 2)})
    except:
        return jsonify({'Erreur': 'Erreur sur homography'})

    start_time = time.time()

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                 LOCALISATION DU LOGO #
    # -------------------------------------------------------------------------------------------------------------------- #
    try:
        obj = np.empty((len(good_matches), 2), dtype=np.float32)
        scene = np.empty((len(good_matches), 2), dtype=np.float32)
        for i in range(len(good_matches)):
            # -- Get the keypoints from the good matches
            obj[i, 0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
        H, _ = cv.findHomography(obj, scene, cv.RANSAC)

        debug.update({"5-LOCALIZE": round(time.time() - start_time, 2)})
    except:
        return jsonify({'Erreur': 'Votre photo ne permet pas de voir le logo'})

    start_time = time.time()

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                    DIMENSION DU LOGO #
    # -------------------------------------------------------------------------------------------------------------------- #
    try:
        obj_corners = np.empty((4, 1, 2), dtype=np.float32)
        obj_corners[0, 0, 0] = 0
        obj_corners[0, 0, 1] = 0
        obj_corners[1, 0, 0] = logo_small.shape[1]
        obj_corners[1, 0, 1] = 0
        obj_corners[2, 0, 0] = logo_small.shape[1]
        obj_corners[2, 0, 1] = logo_small.shape[0]
        obj_corners[3, 0, 0] = 0
        obj_corners[3, 0, 1] = logo_small.shape[0]
        scene_corners = cv.perspectiveTransform(obj_corners, H)
        x_cords = [scene_corners[0, 0, 0], scene_corners[1, 0, 0],
                   scene_corners[2, 0, 0], scene_corners[3, 0, 0], scene_corners[0, 0, 0]]
        y_cords = [scene_corners[0, 0, 1], scene_corners[1, 0, 1],
                   scene_corners[2, 0, 1], scene_corners[3, 0, 1], scene_corners[1, 0, 0]]
        taille_x1 = round(scene_corners[1, 0, 0] - scene_corners[0, 0, 0])
        taille_x2 = round(scene_corners[2, 0, 0] - scene_corners[3, 0, 0])
        taille_y1 = round(scene_corners[2, 0, 1] - scene_corners[0, 0, 1])
        taille_y2 = round(scene_corners[3, 0, 1] - scene_corners[1, 0, 1])
        moyenne_taille_x = round(abs(taille_x1 + taille_x2) / 2)
        moyenne_taille_y = round(abs(taille_y1 + taille_y2) / 2)
        rapport = round(moyenne_taille_x / moyenne_taille_y, 4)
        debug.update(
            {"6-Etiquette taille en pixel x": moyenne_taille_x})
        debug.update(
            {"7-Etiquette taille en pixel y": moyenne_taille_y})
        debug.update(
            {"8-Rapport de taille (idéal:2,6379)": rapport})
       # on arrete le calcul un peu plus loin (pour avoir une photo de l'étiquette) si le rapport est trop différent
        # debug.update({'Dimensions étiquette y': y_cords})
    except:
        return jsonify({'Erreur': 'Le logo est illisible sur votre photo'})

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                         AIRE DU LOGO #
    # -------------------------------------------------------------------------------------------------------------------- #
    try:
        area_etiquette = 0
        for x in range(4-2):
            v1, v2, v3 = 0, x+1, x+2
            tr_area = abs(0.5*(x_cords[v1]*(y_cords[v2]-y_cords[v3]) +
                               x_cords[v2]*(y_cords[v3]-y_cords[v1]) +
                               x_cords[v3]*(y_cords[v1]-y_cords[v2])))
            area_etiquette += tr_area
            debug.update(
                {"9-Etiquette aire en pixel": area_etiquette})
    except:
        return jsonify({'Erreur': 'Aire du logo non mesurable'})

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                    DESSIN DES LIGNES #
    # -------------------------------------------------------------------------------------------------------------------- #
    try:
        photo_couleur = resize(
            cv.imread(destination+'base.jpg', cv.IMREAD_COLOR))
        cv.line(photo_couleur, (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])),
                (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])), (0, 255, 0), 4)
        cv.line(photo_couleur, (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])),
                (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])), (0, 255, 0), 4)
        cv.line(photo_couleur, (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])),
                (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])), (0, 255, 0), 4)
        cv.line(photo_couleur, (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])),
                (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])), (0, 255, 0), 4)
        debug.update({"91-Draw lines": round(time.time() - start_time, 2)})
        start_time = time.time()
        cv.imwrite(dest + '-A)photo_avec_lines_object.jpg', photo_couleur)
        cv.imwrite(repdest + '-A)photo.jpg', photo)
    except:
        return jsonify({'Erreur': 'Dessin des lignes'})

    # arrêt si le rapport est trop différent
    if abs(rapport - 2.6379) > .2:
        # +str(abs(rapport - 2.6379))})
        return jsonify({'Erreur': 'Votre image ne permet pas de voir le logo'})

    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                                                                  SUPPRESSION DU FOND #
    # -------------------------------------------------------------------------------------------------------------------- #

    try:

        # photo_small = cv.cvtColor(photo_small, cv.COLOR_BGR2GRAY)
        photo_small = cv2.GaussianBlur(photo_small, (3, 3), 0)
        start_time = time.time()
        cv.imwrite(dest+'-B)photo_for_remove.jpg',
                   photo_small)  # TODO: bug à trouve
        photo_remove = cv.imread(dest + '-B)photo_for_remove.jpg')
        photo_remove = resize(photo_remove, largrmini)
        # suppresion background
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        iteration = 2
        r = 8
    except:
        return jsonify({'Erreur': 'Le fond de votre image est indétectable'})

    debug.update({"92-Remplissage": round(time.time() - start_time, 2)})
    start_time = time.time()
    try:
        # creation d'une image à la taille de photo_remove
        mask_fore = np.zeros_like(photo_remove)
        debug.update({"debut back": round(time.time() - start_time, 2)})
        start_time = time.time()
        # cv.namedWindow('image')
        x, y, _ = photo_remove.shape
        cv.line(mask_fore, (r, r), (r, x-r), (0, 255, 0), r)
        cv.line(mask_fore, (r, x-r), (y-r, x-r), (0, 255, 0), r)
        cv.line(mask_fore, (y-r, x-r), (y-r, r), (0, 255, 0), r)
        cv.line(mask_fore, (y-r, r), (r, r), (0, 255, 0), r)
        debug.update({"93-lines de selection du fond":
                      round(time.time() - start_time, 2)})
    except:
        return jsonify({'Erreur': 'Création image à la taille de photo remove'})

    start_time = time.time()
    try:
        mask_global = np.zeros(photo_remove.shape[:2], np.uint8) + 2
        mask_global[mask_fore[:, :, 1] == 255] = 1
        debug.update({"93-masks": round(time.time() - start_time, 2)})
    except:
        return jsonify({'Erreur': 'Création du mask'})

    start_time = time.time()
# -------------------------------------------------------------------------------------------------------------------- #
#                                                                                               SUPPRESSION BACKGROUND #
# -------------------------------------------------------------------------------------------------------------------- #
    try:
        mask_global, bgdModel, fgdModel = cv.grabCut(
            photo_remove, mask_global, None, bgdModel, fgdModel, iteration, cv.GC_INIT_WITH_MASK)
        mask_global = np.where((mask_global == 2), 1, 0).astype(
            'uint8')  # 0, 1 pour inverser
        photo_no_back = cv.bitwise_and(
            photo_remove, photo_remove, mask=mask_global)
        # cv.imwrite(dest + '-C)photo_no_back.jpg', resize(photo_no_back, largr))
        image = remove_background(photo_no_back, threshold=250.)
        cv.imwrite(dest + '-C)photo_no_back.png', resize(image, largr))

        debug.update({"94-back retire": round(time.time() - start_time, 2)})
    except:
        return jsonify({'Erreur': 'Le fond est indétectable sur vore photo'})

    start_time = time.time()

# -------------------------------------------------------------------------------------------------------------------- #
#                                                                                                     CALCUL DE L'AIRE #
# -------------------------------------------------------------------------------------------------------------------- #
    try:
        img = cv.imread(dest+'-C)photo_no_back.png', cv.IMREAD_GRAYSCALE)
        area_objet_pixel = cv.countNonZero(img)
        debug.update({"95-calcul de l'aire en pixel":
                      round(time.time() - start_time, 2)})
        debug.update({"95a-Aire en pixel": area_objet_pixel})
    except:
        return jsonify({'Erreur': 'Calcul de l\'aire'})

    start_time = time.time()

# -------------------------------------------------------------------------------------------------------------------- #
#                                                                                        DESSIN DES CONTOURS DE L'AIRE #
# -------------------------------------------------------------------------------------------------------------------- #
    try:
        # load the image, convert it to grayscale, and blur it slightly
        image = cv.imread(dest+'-C)photo_no_back.png')
        edged = cv.Canny(image, 50, 100)
        edged = cv.dilate(edged, None, iterations=1)
        im = cv.erode(edged, None, iterations=1)
        # imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(im, 0, 255, 0)
        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        area = cv.contourArea(contours[0])
        # cv.drawContours(im, contours, -1, (0, 255, 0), 3)
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        largeur = w  # *largr/largrmini
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv.circle(image, (int(x)+int(w), int(y)+int(h)),
                  5, (0, 0, 255), -1)
        taille_objet_x = round(
            w * reel_etiquette_x / moyenne_taille_x/100, 2)
        taille_objet_y = round(h * reel_etiquette_y /
                               moyenne_taille_y / 100, 2)
        debug.update({'96a-Largeur objet en m': taille_objet_x})
        debug.update({'96b-Hauteur objet en m': taille_objet_y})

    except:
        return jsonify({'Erreur': 'Calcul du contour'})
    # try:
    #     # calcul du moment
    #     gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #     ret, thresh = cv.threshold(gray_image, 127, 255, 0)
    #     M = cv.moments(thresh)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     debug.update({'tablo': cX})
    #     # Dessin du moment
    #     cv.circle(image, (int(cX), int(cY)), 5, (0, 0, 255), -1)
    #     debug.update({'x': x, 'y': y, 'w': w, 'h': h,
    #                           'largr': largr, 'largmin': largrmini})
    #     im = cv.imwrite(dest + '-D)photo_no_back_contour.jpg', image)
    #     taille_objet_x = round(w * reel_etiquette_x / moyenne_taille_x/100, 2)
    #     taille_objet_y = round(h * reel_etiquette_y /
    #                            moyenne_taille_y / 100, 2)
    #     taille_objet_c_x = round(
    #         (cX*2)*reel_etiquette_x / moyenne_taille_x/100, 2)
    #     taille_objet_c_y = round(
    #         (cY*2)*reel_etiquette_y / moyenne_taille_y/100, 2)
    #     debug.update(
    #         {'96a-Taille objetx (corrigée) en m': taille_objet_c_x})
    #     debug.update(
    #         {'96b-Taille objetx (corrigée) en m': taille_objet_c_y})

    # except:
    #     return jsonify({'Erreur': 'Calcul du moment'})


# -------------------------------------------------------------------------------------------------------------------- #
#                                                                                ON VÉRIFIE SI C'EST LA DEUXIEME IMAGE #
# -------------------------------------------------------------------------------------------------------------------- #
    try:
        etiquette_area_mm2 = 204 * 84
        data = {}
        context = ({'debug': debug})
        context.update({'resultats': {'area_etiquette': round(area_etiquette, 2),
                                      'area_objet_pixel': round(area_objet_pixel, 2),
                                      'area_objet_m2': round(area_objet_pixel*etiquette_area_mm2 / area_etiquette / 1000000, 2),
                                      'largeur': round(largeur*(etiquette_area_mm2/area_etiquette)/1000, 2)
                                      }})

        if path.exists(destination+id+'-resultats.json'):  # si c'est la deuxieme
            with open(destination+id+'-resultats.json') as json_file:
                data = json.load(json_file)
                volume_pixel = taille_objet_x * \
                    data['photo_1']['resultats']['area_objet_m2']
                data['photo_2'] = context
                data['volume_m3'] = round(volume_pixel, 2)

        else:
            data['photo_1'] = context
        with open(destination+id+'-resultats.json', 'w') as outfile:
            json.dump(data, outfile)

        return jsonify(data)
    except:
        return jsonify({'Erreur': 'Votre photo est trop floue, les résultats ne sont pas lisibles'})


if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    # We now use this syntax to server our app.
    serve(app, host='0.0.0.0', port=8080)

# code pour remettrre l'objet à l'équerre
#   img_before = cv2.imread('base.jpg')
#         img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
#         img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
#         lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
#                                 100, minLineLength=100, maxLineGap=5)

#         angles = []

#         for [[x1, y1, x2, y2]] in lines:
#             cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
#             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             angles.append(angle)

#         median_angle = np.median(angles)
#         debug.update({'1-angle': median_angle})
#         img_rotated = ndimage.rotate(img_before, median_angle)
#         cv2.imwrite(destination+'rotated.jpg', img_rotated)


# code pour détecter si l'objet est tourné
    #    rect = cv.minAreaRect(contours[0])
    #    box = cv.boxPoints(rect)
    #    box = np.int0(box)
    #    cv.drawContours(image, [box], 0, (0, 0, 255), 2)
