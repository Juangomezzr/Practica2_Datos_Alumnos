"""
COMANDOS:

python main_panels.ocr.py  ##Ejecuta el proceso completo de entrenamiento del clasificador LDA+Bayes con data augmentation, evaluación en test_ocr y aplicación a los paneles recortados de test_ocr_panels, generando resultado.txt con las predicciones.

EJERCICIO 1:
python evaluar_clasificadores_OCR.py --classifier lda_bayes  ##Clasificador LDA+Bayes (implementado en ocr_classifier.py)

EJERCICIO 2:
python evaluar_clasificadores_OCR.py --classifier pca_knn  ##Clasificador PCA+KNN
python evaluar_clasificadores_OCR.py --classifier pca_bayes ##Clasificador PCA+Bayes
python evaluar_clasificadores_OCR.py --classifier hog_svm   ##Clasificador HOG+SVM (previamente entrenado con train_hog_svm.py)

EJERICIO 3:
python train_hog_svm.py  ##Entrena el clasificador con los caracteres recortados de train_ocr y guarda el modelo en hog_svm_model.pkl
python evaluar_resultados_test_ocr_panels.py  ##Evalúa el modelo entrenado en los paneles recortados de test_ocr_panels y muestra la precisión, matriz de confusión e histograma de distancia de Levenshtein
python text_detector_OCR.py  ##Detecta paneles y predice caracteres de una imagen especifica.
python ocr_classifier.py  ##Implementa el clasificador LDA+Bayes para caracteres OCR, con métodos para entrenar, predecir y convertir entre caracteres y etiquetas.
"""

import argparse
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import string

from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from hog_svm_classifier import HogSvmClassifier
from text_detector_OCR import nms


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


def detectar_boxes_from_thresh(thresh, h_img, w_img):
    contornos, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ratio = h / float(w) if w > 0 else 0
        if ratio < 0.4 or ratio > 4.5: continue
        if area < 80 or area > 5000: continue
        if w < 5 or h < 8 or w > 100 or h > 120: continue
        margen = 8
        if x < margen or y < margen or (x+w) > (w_img-margen) or (y+h) > (h_img-margen): continue
        roi_t = thresh[y:y+h, x:x+w]
        fill = np.sum(roi_t > 0) / float(area)
        if fill < 0.10 or fill > 0.85: continue
        boxes.append((x, y, w, h))
    return nms(boxes, overlapThresh=0.4)


def detectar_caracteres_panel(img):
    escala = 1.5
    img_scaled = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    thresh_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=11, C=6
    )
    thresh_norm = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=6
    )

    boxes_inv = detectar_boxes_from_thresh(thresh_inv, h_img, w_img)
    boxes_norm = detectar_boxes_from_thresh(thresh_norm, h_img, w_img)

    def score_boxes(boxes):
        if not boxes: return 0
        h_media = np.mean([bh for (bx,by,bw,bh) in boxes])
        return len(boxes) * h_media

    if score_boxes(boxes_norm) > score_boxes(boxes_inv):
        boxes = boxes_norm
        polaridad = 'norm'
    else:
        boxes = boxes_inv
        polaridad = 'inv'

    return boxes, gray, img_scaled, polaridad


# ============================================================
# RANSAC
# ============================================================
def ransac_lineas(boxes, umbral_y=20, min_inliers=1, max_iter=100):
    """
    Agrupa cajas en líneas de texto usando RANSAC iterativo.
    Proceso: encuentra línea → elimina inliers → repite.
    Ordena líneas de arriba a abajo y caracteres de izquierda a derecha.
    """
    if len(boxes) == 0:
        return []

    restantes = list(boxes)
    lineas = []

    while len(restantes) >= min_inliers:
        mejor_inliers = []

        for _ in range(min(max_iter, len(restantes))):
            idx = np.random.randint(0, len(restantes))
            bx, by, bw, bh = restantes[idx]
            cy_ref = by + bh // 2

            inliers = [b for b in restantes
                       if abs((b[1] + b[3] // 2) - cy_ref) < umbral_y]

            if len(inliers) > len(mejor_inliers):
                mejor_inliers = inliers

        if len(mejor_inliers) < min_inliers:
            break

        # Refinar con la media Y de los inliers
        cy_medio = np.mean([b[1] + b[3] // 2 for b in mejor_inliers])
        inliers_final = [b for b in restantes
                         if abs((b[1] + b[3] // 2) - cy_medio) < umbral_y]

        # Ordenar de izquierda a derecha
        lineas.append(sorted(inliers_final, key=lambda b: b[0]))

        # Eliminar inliers de restantes
        restantes = [b for b in restantes if b not in inliers_final]

    # Ordenar líneas de arriba a abajo
    lineas.sort(key=lambda l: np.mean([b[1] for b in l]))
    return lineas


# ============================================================
#   Clasificar cada carácter y construir el texto
# ============================================================
def procesar_panel_ocr(img_path, clf):
    img = cv2.imread(img_path)
    if img is None:
        return "", 0, 0

    h_orig, w_orig = img.shape[:2]

    # detectar caracteres
    boxes, gray, img_scaled, polaridad = detectar_caracteres_panel(img)

    #  agrupar en líneas con RANSAC
    # umbral_y proporcional al tamaño típico de letra
    alturas = [h for (x, y, w, h) in boxes] if boxes else [20]
    h_media = np.median(alturas) if alturas else 20
    umbral_y = max(15, int(h_media * 0.6))
    lineas = ransac_lineas(boxes, umbral_y=umbral_y, min_inliers=1)

    # clasificar en orden izquierda-derecha, arriba-abajo
    texto_lineas = []
    for linea in lineas:
        texto_linea = ""
        for (x, y, w, h) in linea:
            
            roi_bgr = img_scaled[y:y+h, x:x+w]
            if roi_bgr.size == 0:
                continue

            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            roi_gray = clahe.apply(roi_gray)
            if polaridad == 'norm':
                _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            roi_final = cv2.cvtColor(roi_bin, cv2.COLOR_GRAY2BGR)
            pred = clf.predict(roi_final)    
            char = clf.label2char(pred)
            texto_linea += char

        if texto_linea:
            texto_lineas.append(texto_linea)

    # Formato: líneas separadas por '+', sin espacios
    return "+".join(texto_lineas), w_orig, h_orig


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, nargs="?", default="LdaNormalBayes")
    parser.add_argument('--train_path', default="train_ocr")
    parser.add_argument('--test_path', default="test_ocr")
    parser.add_argument('--panels_path', default="test_ocr_panels")
    parser.add_argument('--output', default="test_ocr_panels/resultado.txt")
    args = parser.parse_args()

    # ── Entrenar LDA+Bayes ───────────────────────────────────────
    print(f"Cargando datos de entrenamiento desde: {args.train_path}")
    train_dict = cargar_diccionario_imagenes(args.train_path)
    print(f"  Clases encontradas: {len(train_dict)}")

    print("Entrenando clasificador LDA+Bayes...")
    ocr_classifier = LdaNormalBayesClassifier(ocr_char_size=(25, 25))
    
    train_dict_aug = {}
    for key, imgs in train_dict.items():
        augmented = list(imgs)
        for img in imgs[:50]:  # augmentar solo 50 por clase para no tardar demasiado
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
            inv = cv2.bitwise_not(gray)
            inv_bgr = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
            augmented.append(inv_bgr)
        train_dict_aug[key] = augmented

    ocr_classifier.train(train_dict_aug)

    # ── Evaluar en test_ocr ──────────────────────────────────────
    print(f"\nCargando datos de validación desde: {args.test_path}")
    test_dict = cargar_diccionario_imagenes(args.test_path)

    print("Iniciando evaluación de validación...")
    true_labels = ocr_classifier.get_labels_dict(test_dict)
    predicted_labels, mean_pred_time = ocr_classifier.predict_dict(test_dict)

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    cm = confusion_matrix(true_labels, predicted_labels)
    classes = list('0123456789' + string.ascii_letters)
    plot_confusion_matrix(cm, classes, args.detector)

    # ── OCR paneles → resultado.txt ─────────────────
    print("\n" + "="*40)
    print("EJERCICIO 3: OCR de paneles recortados")
    print("="*40)

    clf = ocr_classifier  # LDA+Bayes, mejor clasificador

    imagenes_panel = sorted(
        glob.glob(os.path.join(args.panels_path, "*.png")) +
        glob.glob(os.path.join(args.panels_path, "*.jpg"))
    )
    print(f"Paneles encontrados: {len(imagenes_panel)}")

    resultados = []
    for img_path in imagenes_panel:
        nombre = os.path.basename(img_path)
        texto, w, h = procesar_panel_ocr(img_path, clf)
        print(f"  {nombre} -> \"{texto}\"")
        resultados.append(f"{nombre};0;0;{w};{h};;;{texto}")

    with open(args.output, "w") as f:
        for linea in resultados:
            f.write(linea + "\n")

    print(f"\nresultado.txt guardado en: {args.output}")
    print(f"Total paneles procesados: {len(resultados)}")