"""
COMANDOS:
python main.py --test_path Practica1/test_detection --train_ocr train_ocr  ## Ejecuta todo el pipeline sin visualización
python main.py --test_path Practica1/test_detection --train_ocr train_ocr --visualize_ocr  ## Ejecuta el pipeline mostrando visualización OCR 
"""


import argparse
import os
import glob
import cv2
import numpy as np
from pathlib import Path

# ── Detector MSER (Práctica 1) ────────────────────────────────────────────────
from Practica1.detectors.detector_mser import (
    crear_detector_mser,
    crear_configuracion_color,
    preprocesar_imagen,
    obtener_candidatos,
    filtrar_detecciones_por_color,
    nms_maximos_locales,
)

# ── OCR (Práctica 2) ──────────────────────────────────────────────────────────
from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from main_panels_ocr import (
    cargar_diccionario_imagenes,
    detectar_caracteres_panel,
    ransac_lineas,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Entrenar clasificador LDA+Bayes
# ══════════════════════════════════════════════════════════════════════════════
def entrenar_clasificador(train_path="train_ocr"):
    print("Cargando datos de entrenamiento...")
    train_dict = cargar_diccionario_imagenes(train_path)
    print(f"  Clases encontradas: {len(train_dict)}")

    print("Entrenando clasificador LDA+Bayes...")
    clf = LdaNormalBayesClassifier(ocr_char_size=(25, 25))

    # Data augmentation (igual que en main_panels_ocr.py)
    train_dict_aug = {}
    for key, imgs in train_dict.items():
        augmented = list(imgs)
        for img in imgs[:50]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            inv = cv2.bitwise_not(gray)
            inv_bgr = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
            augmented.append(inv_bgr)
        train_dict_aug[key] = augmented

    clf.train(train_dict_aug)
    print("Clasificador entrenado.\n")
    return clf


# ══════════════════════════════════════════════════════════════════════════════
#  OCR sobre un recorte de panel (imagen BGR ya cargada, no path)
# ══════════════════════════════════════════════════════════════════════════════
def ocr_sobre_recorte(img_bgr, clf):
    if img_bgr is None or img_bgr.size == 0:
        return "", [], None, []

    boxes, gray, img_scaled, polaridad = detectar_caracteres_panel(img_bgr)

    alturas = [h for (x, y, w, h) in boxes] if boxes else [20]
    h_media = np.median(alturas) if alturas else 20
    umbral_y = max(15, int(h_media * 0.6))
    lineas = ransac_lineas(boxes, umbral_y=umbral_y, min_inliers=1)

    texto_lineas = []
    # lineas_con_chars: lista de listas de (box, char)
    lineas_con_chars = []

    for linea in lineas:
        texto_linea = ""
        linea_chars = []
        for (x, y, w, h) in linea:
            roi_bgr = img_scaled[y:y+h, x:x+w]
            if roi_bgr.size == 0:
                linea_chars.append(((x, y, w, h), "?"))
                continue
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            roi_gray = clahe.apply(roi_gray)
            if polaridad == 'norm':
                _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            roi_final = cv2.cvtColor(roi_bin, cv2.COLOR_GRAY2BGR)
            pred = clf.predict(roi_final)
            char = clf.label2char(pred)
            texto_linea += char
            linea_chars.append(((x, y, w, h), char))

        if texto_linea:
            texto_lineas.append(texto_linea)
        lineas_con_chars.append(linea_chars)

    return "+".join(texto_lineas), lineas, img_scaled, lineas_con_chars


def visualizar_resultado(img_original, detecciones_finales, resultados_ocr, nombre_img):
    vis = img_original.copy()
    escala = 1.5

    for (x1, y1, x2, y2, score), (texto, lineas, img_scaled, lineas_con_chars) in zip(detecciones_finales, resultados_ocr):

        # 1. Rectángulo del panel + score
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if img_scaled is None or not lineas_con_chars:
            continue

        for linea_chars in lineas_con_chars:
            centros = []

    # Calcular altura media de la línea para que la recta sea horizontal
            cy_medio = int(np.mean([
            y1 + int((by + bh // 2) / escala)
        for (bx, by, bw, bh), char in linea_chars
        ]))

        for (bx, by, bw, bh), char in linea_chars:
            bx_orig = x1 + int(bx / escala)
            by_orig = y1 + int(by / escala)
            bw_orig = max(1, int(bw / escala))
            bh_orig = max(1, int(bh / escala))

        # Rectángulo sobre cada carácter
            cv2.rectangle(vis,
                        (bx_orig, by_orig),
                        (bx_orig + bw_orig, by_orig + bh_orig),
                        (255, 0, 0), 1)

        # Carácter reconocido encima de su rectángulo
            cv2.putText(vis, char,
                        (bx_orig, max(10, by_orig - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 255), 1, cv2.LINE_AA)

            cx = bx_orig + bw_orig // 2
            centros.append((cx, cy_medio))  

    # Línea recta horizontal que une los caracteres
        for i in range(len(centros) - 1):
            cv2.line(vis, centros[i], centros[i + 1], (0, 165, 255), 2)

    cv2.imshow(f"Resultado: {nombre_img}", vis)
    print(f"  Pulsa cualquier tecla para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════
def detectar_y_leer_paneles(test_path, clf, visualize_ocr=False):
    os.makedirs("Practica1/resultado_imgs", exist_ok=True)
    archivo_resultados = open("Practica1/resultado.txt", "w")

    mser = crear_detector_mser()
    config_color = crear_configuracion_color()

    imagenes = sorted(Path(test_path).glob("*.png"))
    print(f"Procesando {len(imagenes)} imágenes en: {test_path}\n")

    for img_path in imagenes:
        nombre = img_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] No se pudo cargar {nombre}, saltando.")
            continue

        print(f"Procesando: {nombre}")

        # ── Detección de paneles (MSER) ───────────────────────────────────
        _, eq = preprocesar_imagen(img)
        _, candidatos = obtener_candidatos(mser, eq)
        detecciones = filtrar_detecciones_por_color(img, candidatos, config_color)
        detecciones_finales = nms_maximos_locales(detecciones)
        print(f"  Paneles detectados: {len(detecciones_finales)}")

        # ── OCR sobre cada panel ──────────────────────────────────────────
        resultados_ocr = []
        for (x1, y1, x2, y2, score) in detecciones_finales:
            recorte = img[y1:y2, x1:x2]
            texto, lineas, img_scaled, lineas_con_chars = ocr_sobre_recorte(recorte, clf)
            resultados_ocr.append((texto, lineas, img_scaled, lineas_con_chars))
            print(f"    Panel ({x1},{y1})-({x2},{y2}) score={score:.2f} -> \"{texto}\"")

            # Escribir resultado
            linea = f"{nombre};{x1};{y1};{x2};{y2};1;{score:.4f};{texto}\n"
            archivo_resultados.write(linea)

        # ── Guardar imagen con detecciones ────────────────────────────────
        img_guardada = img.copy()
        for (x1, y1, x2, y2, score) in detecciones_finales:
            cv2.rectangle(img_guardada, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_guardada, f"{score:.2f}", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join("Practica1/resultado_imgs", nombre), img_guardada)

        # ── Visualización opcional ────────────────────────────────────────
        if visualize_ocr:
            visualizar_resultado(img, detecciones_finales, resultados_ocr, nombre)

    archivo_resultados.close()
    print("\nFinalizado. Resultados en 'resultado.txt' y 'resultado_imgs/'")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detector de paneles MSER + OCR integrado"
    )
    parser.add_argument("--detector", type=str, default="mser")
    parser.add_argument("--train_path", default="train_detection")
    parser.add_argument("--test_path",  default="Practica1/test_detection")
    parser.add_argument("--train_ocr",  default="train_ocr")
    parser.add_argument("--visualize_ocr", action="store_true", default=False,
                        help="Mostrar visualización OCR (por defecto False)")
    args = parser.parse_args()

    # 1. Entrenar clasificador OCR
    clf = entrenar_clasificador(args.train_ocr)

    # 2. Detectar paneles y leer texto
    detectar_y_leer_paneles(args.test_path, clf, visualize_ocr=args.visualize_ocr)