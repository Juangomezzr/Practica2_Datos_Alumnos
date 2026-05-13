"""
train_hog_svm.py  –  VERSIÓN CORREGIDA
=======================================
FIX PRINCIPAL: el modelo se entrena aplicando exactamente el mismo
preprocesado que se aplica al ROI durante la predicción en paneles.

Antes el modelo veía imágenes limpias 62x62 en train y ROIs con
OTSU+blur+equalizeHist+borde en predict  → dominio completamente distinto.
Ahora train y predict ven exactamente lo mismo.
"""

import cv2
import os
import numpy as np
from skimage.feature import hog
from hog_svm_classifier import HogSvmClassifier

TRAIN_PATH = "train_ocr"
MAX_IMGS   = 300   # más muestras que antes (150) → mejor generalización


# ──────────────────────────────────────────────────────────────
#  PREPROCESADO CANÓNICO  (idéntico al de procesar_panel_ocr)
# ──────────────────────────────────────────────────────────────
def preprocesar_roi(img, size=(25, 25)):
    """
    Convierte cualquier ROI (color o gris) al mismo espacio de
    características que se usa durante la predicción en paneles:
      1. Escala de grises
      2. Resize a 25x25
      3. Binarización Otsu
      4. GaussianBlur (suaviza ruido de binarización)
      5. EqualizeHist  (normaliza contraste)
      6. Borde negro 2px  → 29x29
    El _hog_features del clasificador hará el resize final a 25x25.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.equalizeHist(img)
    img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    return img   # 29x29


def cargar_dataset(path):
    images_dict = {}

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue

        # Caso 1: dígitos 0–9 (carpeta directa)
        if folder.isdigit():
            key = folder
            images_dict[key] = []
            filenames = sorted(os.listdir(folder_path))[:MAX_IMGS]
            for filename in filenames:
                img = cv2.imread(os.path.join(folder_path, filename))
                if img is not None:
                    images_dict[key].append(preprocesar_roi(img))

        # Caso 2: mayúsculas / minúsculas (subcarpeta por letra)
        elif folder in ("may", "min"):
            for sub in os.listdir(folder_path):
                sub_path = os.path.join(folder_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                key = sub
                images_dict[key] = []
                filenames = sorted(os.listdir(sub_path))[:MAX_IMGS]
                for filename in filenames:
                    img = cv2.imread(os.path.join(sub_path, filename))
                    if img is not None:
                        images_dict[key].append(preprocesar_roi(img))

    return images_dict


if __name__ == "__main__":
    print("Cargando dataset con preprocesado canónico...")
    train_dict = cargar_dataset(TRAIN_PATH)

    print("Resumen del dataset:")
    for k, v in sorted(train_dict.items()):
        print(f"  {k}: {len(v)} imágenes")

    print("\nEntrenando HOG+SVM (kernel LINEAR, C=10)...")
    clf = HogSvmClassifier()
    clf.svm.setKernel(cv2.ml.SVM_LINEAR)
    clf.svm.setC(10.0)          # C más alto que por defecto → mejor accuracy
    clf.svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

    clf.train(train_dict)

    print("Guardando modelo...")
    clf.save("hog_svm_model.xml")
    print("Modelo guardado como hog_svm_model.xml")