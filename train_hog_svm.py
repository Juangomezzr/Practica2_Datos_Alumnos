import cv2
import os
from hog_svm_classifier import HogSvmClassifier

TRAIN_PATH = "train_ocr"
MAX_IMGS = 150   # limitar imágenes por clase para evitar colapso

def cargar_dataset(path):
    images_dict = {}

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)

        if not os.path.isdir(folder_path):
            continue

        # Caso 1: dígitos 0–9
        if folder.isdigit():
            key = folder
            images_dict[key] = []

            for filename in os.listdir(folder_path)[:MAX_IMGS]:
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images_dict[key].append(img)

        # Caso 2: mayúsculas y minúsculas
        elif folder in ["may", "min"]:
            for sub in os.listdir(folder_path):
                sub_path = os.path.join(folder_path, sub)
                if not os.path.isdir(sub_path):
                    continue

                key = sub
                images_dict[key] = []

                for filename in os.listdir(sub_path)[:MAX_IMGS]:
                    img_path = os.path.join(sub_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images_dict[key].append(img)

    return images_dict


if __name__ == "__main__":
    print("Cargando dataset...")
    train_dict = cargar_dataset(TRAIN_PATH)

    print("Resumen del dataset:")
    for k, v in train_dict.items():
        print(k, len(v))

    print("\nEntrenando HOG+SVM (LINEAL, RÁPIDO)...")
    clf = HogSvmClassifier()

    # CAMBIO IMPORTANTE: kernel lineal
    clf.svm.setKernel(cv2.ml.SVM_LINEAR)

    clf.train(train_dict)

    print("Guardando modelo...")
    clf.save("hog_svm_model.xml")

    print("Modelo guardado como hog_svm_model.xml")
