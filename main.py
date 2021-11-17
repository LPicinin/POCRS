import cv2
import numpy as np
import imutils
import streamlit as st

from utils import encontrar_contornos, ordenar_pontos

st.title('POCRS')
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)
modo = st.radio('Exibir', ('Normal', 'Cinza', 'Borrado', 'Bordas', 'Maior Contorno', 'Rotacioar imagem'))

while run:
    ret, frame = cam.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (H, W) = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 60, 160)
    conts, maior = encontrar_contornos(edged)
    if modo == 'Cinza':
        img = gray
    elif modo == 'Borrado':
        img = blur
    elif modo == 'Bordas':
        img = edged
    elif modo == 'Maior Contorno' and maior is not None:
        cv2.drawContours(img, maior, -1, (120, 255, 0), 28)
        cv2.drawContours(img, [maior], -1, (120, 255, 0), 2)
    elif modo == 'Rotacioar imagem' and maior is not None:
        pontos_maior = ordenar_pontos(maior)
        pts1 = np.float32(pontos_maior)
        pts2 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        matriz = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, matriz, (W, H))
    FRAME_WINDOW.image(img)
if not run:
    st.write('Parado')
