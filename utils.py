import cv2
import imutils
import numpy as np


def encontrar_contornos(img):
    conts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key=cv2.contourArea, reverse=True)[:6]
    maior = None
    for c in conts:
        perimetro = cv2.arcLength(c, True)
        aproximacao = cv2.approxPolyDP(c, 0.02 * perimetro, True)
        if len(aproximacao) == 4:
            maior = aproximacao
            break

    return conts, maior


def ordenar_pontos(pontos):
    pontos = pontos.reshape((4, 2))
    pontos_novos = np.zeros((4, 1, 2), dtype=np.int32)

    add = pontos.sum(1)
    pontos_novos[0] = pontos[np.argmin(add)]
    pontos_novos[2] = pontos[np.argmax(add)]

    dif = np.diff(pontos, axis=1)
    pontos_novos[1] = pontos[np.argmin(dif)]
    pontos_novos[3] = pontos[np.argmax(dif)]

    return pontos_novos
