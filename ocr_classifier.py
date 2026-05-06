# @brief OCRClassifier
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025
#

from http.client import responses
import string
import time
import cv2
from matplotlib.pyplot import gray
import numpy as np


class OCRClassifier:
    """
    Classifier for Optical Character Recognition
    """

    def __init__(self, ocr_char_size=(25, 25)):
        self.ocr_char_size = ocr_char_size
        self.classifier_name = None


    def char2label(self, c):
        all_chars = '0123456789' + string.ascii_letters
        return all_chars.find(c)


    def label2char(self, label):
        all_chars = '0123456789' + string.ascii_letters
        return all_chars[label]

    def get_labels_dict(self, images_dict):
        responses = []
        for key in sorted(images_dict.keys()):
          for img in images_dict[key]:
            responses.append(self.char2label(key))

        return np.array(responses, dtype=np.int32)
    def predict_dict(self, images_dict):
        responses = []
        tiempos = []

        for key in sorted(images_dict.keys()):
            for img in images_dict[key]:
                t0 = time.time()
                pred = self.predict(img)
                tiempos.append((time.time() - t0) * 1000.0)  # ms
                responses.append(pred)

        mean_pred_time = sum(tiempos) / len(tiempos) if len(tiempos) > 0 else 0.0

        print(f" Tiempo medio predict(): {mean_pred_time:.3f} ms")
        print(f" Tiempo máximo predict(): {max(tiempos):.3f} ms")
        print(f" Tiempo mínimo predict(): {min(tiempos):.3f} ms")

        return np.array(responses, dtype=np.int32), mean_pred_time

    def _extraer_caracteristicas(self, img):
    # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Umbralizar
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Buscar contornos
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
        # Imagen vacía → devolver vector de ceros
            return np.zeros((self.ocr_char_size[0] * self.ocr_char_size[1],), dtype=np.float32)

    # Tomar el contorno más grande
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        char_img = th[y:y+h, x:x+w]

    # Redimensionar al tamaño estándar
        char_img = cv2.resize(char_img, self.ocr_char_size)

    # Aplanar a vector 1D
        feat = char_img.flatten().astype(np.float32)

        return feat





