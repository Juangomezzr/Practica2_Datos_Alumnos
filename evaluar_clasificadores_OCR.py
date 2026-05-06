import argparse
import os
import cv2
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import string

from lda_normal_bayes_classifier import LdaNormalBayesClassifier


# ---------------------------------------------------------
# CARGA DE IMÁGENES
# ---------------------------------------------------------
def cargar_diccionario_imagenes(ruta_directorio):
    images_dict = {}

    if not os.path.exists(ruta_directorio):
        print("ERROR: Ruta no encontrada:", ruta_directorio)
        return images_dict

    for nombre_carpeta in sorted(os.listdir(ruta_directorio)):
        ruta_carpeta = os.path.join(ruta_directorio, nombre_carpeta)

        if not os.path.isdir(ruta_carpeta):
            continue

        # Buscar imágenes directamente
        imagenes = sorted(glob.glob(os.path.join(ruta_carpeta, "*.png")))
        if imagenes:
            images_dict[nombre_carpeta] = [cv2.imread(p) for p in imagenes]
            continue

        # Buscar subcarpetas (may/min)
        for subcarpeta in sorted(os.listdir(ruta_carpeta)):
            ruta_sub = os.path.join(ruta_carpeta, subcarpeta)
            if os.path.isdir(ruta_sub):
                imgs = sorted(glob.glob(os.path.join(ruta_sub, "*.png")))
                if imgs:
                    images_dict[subcarpeta] = [cv2.imread(p) for p in imgs]

    return images_dict


# ---------------------------------------------------------
# MATRIZ DE CONFUSIÓN MEJORADA
# ---------------------------------------------------------
def plot_confusion_matrix(cm, classes, classifier_name):
    plt.figure(figsize=(22, 22))

    # Normalizar por filas
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




# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and evaluates an OCR classifier')
    parser.add_argument('--classifier', type=str, default="lda_bayes",
                        help='Classifier: lda_bayes, pca_knn, pca_bayes, hog_svm')
    parser.add_argument('--train_path', default="./train_ocr",
                        help='Training data directory')
    parser.add_argument('--validation_path', default="./test_ocr",
                        help='Validation data directory')

    args = parser.parse_args()

    # 1) Cargar imágenes de entrenamiento
    print("Cargando datos de entrenamiento...")
    train_dict = cargar_diccionario_imagenes(args.train_path)
    print(f"  Clases encontradas: {len(train_dict)}")

    # 2) Cargar imágenes de validación
    print("Cargando datos de validación...")
    val_dict = cargar_diccionario_imagenes(args.validation_path)

    # 3) Crear y entrenar clasificador
    if args.classifier == "lda_bayes":
        clf = LdaNormalBayesClassifier(ocr_char_size=(25, 25))

    elif args.classifier == "pca_knn":
        from pca_knn_classifier import PcaKnnClassifier
        clf = PcaKnnClassifier(ocr_char_size=(25, 25))

    elif args.classifier == "pca_bayes":
        from pca_bayes_classifier import PcaBayesClassifier
        clf = PcaBayesClassifier(ocr_char_size=(25, 25))

    elif args.classifier == "hog_svm":
        from hog_svm_classifier import HogSvmClassifier
        clf = HogSvmClassifier(ocr_char_size=(25, 25))

    else:
        print(f"Clasificador '{args.classifier}' no reconocido. Usando lda_bayes.")
        clf = LdaNormalBayesClassifier(ocr_char_size=(25, 25))

    print(f"Entrenando clasificador '{args.classifier}'...")

    t0 = time.time()
    clf.train(train_dict)
    t_train = time.time() - t0
    print(f" Tiempo de entrenamiento: {t_train:.3f} segundos")

    # 4) Predecir sobre validación
    print("Evaluando sobre validación...")
    true_labels = clf.get_labels_dict(val_dict)
    predicted_labels, mean_pred_time = clf.predict_dict(val_dict)

    # 5) Métricas
    acc = accuracy_score(true_labels, predicted_labels)
    print(f"\nAccuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(true_labels, predicted_labels)

    # Lista de clases
    classes = list('0123456789' + string.ascii_letters)

    # Matriz de confusión
    plot_confusion_matrix(cm, classes, args.classifier)

    # Accuracy plot
    plot_accuracy(acc, args.classifier)

    print("\nTiempo medio de predicción (ms):", mean_pred_time)
    print("Tiempo de entrenamiento (s):", t_train)
