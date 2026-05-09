import argparse
import os
import cv2
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from hog_svm_classifier import HogSvmClassifier
# Importar funciones del detector de texto (ejercicio 3)
from text_detector_OCR import detectar_caracteres, nms, agrupar_por_lineas
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


# ============================================================
#  EJERCICIO 3 - RANSAC iterativo para alinear líneas de texto
# ============================================================
def ransac_lineas(boxes, umbral_y=15, min_inliers=1, max_iter=50):
    """
    Agrupa boxes en líneas usando RANSAC sobre el centro Y de cada caja.
    Proceso iterativo: encuentra línea, elimina inliers, repite.
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

            inliers = [b for b in restantes if abs((b[1] + b[3] // 2) - cy_ref) < umbral_y]

            if len(inliers) > len(mejor_inliers):
                mejor_inliers = inliers

        if len(mejor_inliers) < min_inliers:
            break

        # Refinar con Y medio
        cy_medio = np.mean([b[1] + b[3] // 2 for b in mejor_inliers])
        inliers_final = [b for b in restantes
                         if abs((b[1] + b[3] // 2) - cy_medio) < umbral_y]

        lineas.append(sorted(inliers_final, key=lambda b: b[0]))  # izq a der
        restantes = [b for b in restantes if b not in inliers_final]

    # Ordenar líneas de arriba a abajo
    lineas.sort(key=lambda l: np.mean([b[1] for b in l]))
    return lineas


# ============================================================
#  EJERCICIO 3 - Filtros anti-símbolo
# ============================================================
def es_caracter_valido(roi, w, h):
    ratio = h / float(w)
    area = w * h

    if ratio < 0.7 or ratio > 3.5:
        return False
    if area < 140 or area > 3000:
        return False
    if w > 60 or h > 60:
        return False

    edges_roi = cv2.Canny(roi, 40, 120)
    if np.sum(edges_roi > 0) > 450:
        return False

    conts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(conts) == 0:
        return False
    cnt = conts[0]
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) > 10 or cv2.arcLength(cnt, True) > 300:
        return False

    return True


# ============================================================
#  EJERCICIO 3 - Procesar un panel recortado
# ============================================================
def procesar_panel_ocr(img_path, clf):
    img = cv2.imread(img_path)
    if img is None:
        return "", 0, 0

    h_orig, w_orig = img.shape[:2]

    # Reutiliza detectar_caracteres de text_detector_OCR.py
    boxes, img_scaled, gray = detectar_caracteres(img)

    # RANSAC para encontrar líneas (pedido en el enunciado)
    lineas = ransac_lineas(boxes, umbral_y=15, min_inliers=1)

    texto_lineas = []
    for linea in lineas:
        texto_linea = ""
        for (x, y, w, h) in linea:
            roi = gray[y:y+h, x:x+w]

            if not es_caracter_valido(roi, w, h):
                continue

            # Mismo preprocesado que en text_detector_OCR
            roi = cv2.resize(roi, (25, 25), interpolation=cv2.INTER_AREA)
            roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            roi = cv2.GaussianBlur(roi, (3, 3), 0)
            roi = cv2.equalizeHist(roi)
            roi = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

            pred = clf.predict(roi)
            char = clf.label2char(pred)
            texto_linea += char

        if texto_linea:
            texto_lineas.append(texto_linea)

    # Formato del enunciado: líneas separadas por '+', sin espacios
    return "+".join(texto_lineas), w_orig, h_orig


# -----------------------------
# MAIN  (no modificado salvo añadir ejercicio 3 al final)
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
    # Nuevos argumentos para ejercicio 3
    parser.add_argument('--panels_path', default="test_ocr_panels",
                        help='Directorio con los paneles recortados (ejercicio 3)')
    parser.add_argument('--model', default="hog_svm_model.xml",
                        help='Modelo HOG+SVM para OCR de paneles')
    parser.add_argument('--output', default="resultado.txt",
                        help='Fichero de salida resultado.txt (ejercicio 3)')

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

    cm = confusion_matrix(true_labels, predicted_labels)
    classes = list('0123456789' + string.ascii_letters)
    plot_confusion_matrix(cm, classes, args.detector)

    # ============================================================
    #  EJERCICIO 3: OCR paneles recortados -> resultado.txt
    # ============================================================
    print("\n" + "="*40)
    print("EJERCICIO 3: OCR de paneles recortados")
    print("="*40)

    clf = HogSvmClassifier()
    clf.load(args.model)
    print(f"Modelo cargado: {args.model}")

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
        # Formato pedido: <imagen>;<x1>;<y1>;<x2>;<y2>;<label>;<texto_ocr>
        resultados.append(f"{nombre};0;0;{w};{h};;{texto}")

    with open(args.output, "w") as f:
        for linea in resultados:
            f.write(linea + "\n")

    print(f"\nresultado.txt guardado en: {args.output}")
    print(f"Total paneles procesados: {len(resultados)}")