import argparse
import os
import cv2
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from lda_normal_bayes_classifier import LdaNormalBayesClassifier
import string

##python evaluar_clasificadores_OCR.py --classifier lda_bayes
##python evaluar_clasificadores_OCR.py --classifier pca_knn
##python evaluar_clasificadores_OCR.py --classifier pca_bayes
##python evaluar_clasificadores_OCR.py --classifier hog_svm
def cargar_diccionario_imagenes(ruta_directorio):
    images_dict = {}

    if not os.path.exists(ruta_directorio):
        print("ERROR: Ruta no encontrada:", ruta_directorio)
        return images_dict

    for nombre_carpeta in sorted(os.listdir(ruta_directorio)):
        ruta_carpeta = os.path.join(ruta_directorio, nombre_carpeta)

        if not os.path.isdir(ruta_carpeta):
            continue

        imagenes = sorted(glob.glob(os.path.join(ruta_carpeta, "*.png")))
        if imagenes:
            images_dict[nombre_carpeta] = [cv2.imread(p) for p in imagenes]
            continue

        for subcarpeta in sorted(os.listdir(ruta_carpeta)):
            ruta_sub = os.path.join(ruta_carpeta, subcarpeta)
            if os.path.isdir(ruta_sub):
                imgs = sorted(glob.glob(os.path.join(ruta_sub, "*.png")))
                if imgs:
                    images_dict[subcarpeta] = [cv2.imread(p) for p in imgs]

    return images_dict


# -----------------------------
# MATRIZ DE CONFUSIÓN MEJORADA
# -----------------------------
def plot_confusion_matrix(cm, classes, classifier_name):
    plt.figure(figsize=(22, 22))

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm_norm, interpolation='nearest', cmap='viridis')
    plt.title(f"Confusion Matrix - {classifier_name}", fontsize=20)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    print("Matriz de confusión guardada en confusion_matrix.png")




# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument('--detector', type=str, nargs="?", default="LdaNormalBayes",
                        help='Detector string name')
    parser.add_argument('--train_path', default="train_ocr",
                        help='Select the training data dir')
    parser.add_argument('--test_path', default="test_ocr",
                        help='Select the testing data dir')

    args = parser.parse_args()

    print(f"Cargando datos de entrenamiento desde: {args.train_path}")
    train_dict = cargar_diccionario_imagenes(args.train_path)
    print(f"  Clases encontradas: {len(train_dict)}")

    print(f"Instanciando clasificador {args.detector}...")
    ocr_classifier = LdaNormalBayesClassifier(ocr_char_size=(25, 25))

    print("Entrenando clasificador...")
    ocr_classifier.train(train_dict)

    print(f"\nCargando datos de validación desde: {args.test_path}")
    test_dict = cargar_diccionario_imagenes(args.test_path)

    print("\nResumen del conjunto de validación:")
    for k in sorted(test_dict.keys()):
        print(f"  Clase {k}: {len(test_dict[k])} imágenes")
    print(f"Total imágenes de validación: {sum(len(v) for v in test_dict.values())}")

    print("Iniciando evaluación de validación...")

    true_labels = ocr_classifier.get_labels_dict(test_dict)

    t0 = time.time()
    predicted_labels, mean_pred_time = ocr_classifier.predict_dict(test_dict)
    t_pred_total = time.time() - t0

    num_chars = len(predicted_labels)
    print(f" Tiempo medio por carácter: {mean_pred_time:.3f} ms")
    print(f" Tiempo total de predicción: {t_pred_total:.3f} segundos")

    accuracy = accuracy_score(true_labels, predicted_labels)

    print("\n" + "="*40)
    print("📊 RESULTADOS DE LA EVALUACIÓN OCR")
    print("="*40)
    print(f"Total caracteres testeados: {len(true_labels)}")
    print(f"Tasa de Acierto (Accuracy): {accuracy * 100:.2f}%")
    print("="*40)

    # -----------------------------
    # MATRIZ DE CONFUSIÓN + ACCURACY
    # -----------------------------
    cm = confusion_matrix(true_labels, predicted_labels)

    # Lista de clases en orden correcto
    classes = list('0123456789' + string.ascii_letters)

    # Matriz de confusión mejorada
    plot_confusion_matrix(cm, classes, args.detector)

    # Gráfico de accuracy
    plot_accuracy(accuracy, args.detector)
   

