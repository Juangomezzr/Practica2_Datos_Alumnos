import cv2
import numpy as np
from hog_svm_classifier import HogSvmClassifier
##python text_detector_OCR.py


# ============================================================
#  NON-MAXIMUM SUPPRESSION (evita cajas superpuestas)
# ============================================================
def nms(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes)
    x1 = boxes_np[:,0]
    y1 = boxes_np[:,1]
    x2 = boxes_np[:,0] + boxes_np[:,2]
    y2 = boxes_np[:,1] + boxes_np[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Ordenar por ÁREA descendente (más grande primero)
    idxs = np.argsort(areas)[::-1]

    pick = []
    while len(idxs) > 0:
        last = idxs[0]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[1:]])
        yy1 = np.maximum(y1[last], y1[idxs[1:]])
        xx2 = np.minimum(x2[last], x2[idxs[1:]])
        yy2 = np.minimum(y2[last], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[idxs[1:]]

        idxs = np.delete(idxs, np.concatenate(
            ([0], np.where(overlap > overlapThresh)[0] + 1)))

    return [boxes[i] for i in pick]


# ============================================================
#  AGRUPAR CAJAS POR LÍNEAS
# ============================================================
def agrupar_por_lineas(boxes, umbral=25):
    lineas = []
    boxes_sorted = sorted(boxes, key=lambda b: b[1])

    for box in boxes_sorted:
        x, y, w, h = box
        asignado = False

        for linea in lineas:
            if abs(linea[0][1] - y) < umbral:
                linea.append(box)
                asignado = True
                break

        if not asignado:
            lineas.append([box])

    for linea in lineas:
        linea.sort(key=lambda b: b[0])

    return lineas


# ============================================================
#  DETECTOR (MSER + Canny + NMS)
# ============================================================
def detectar_caracteres(img):
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- MSER ----
    mser = cv2.MSER_create()
    mser.setMinArea(60)
    mser.setMaxArea(5000)
    regiones, _ = mser.detectRegions(gray)
    boxes = []
    for p in regiones:
        x, y, w, h = cv2.boundingRect(p.reshape(-1,1,2))
        area = w*h
        ratio = h / float(w)
        if 80 < area < 5000 and 0.90 < ratio < 5.0 and 5 < w < 70 and 12 < h < 100:
            boxes.append((x, y, w, h))

            print(f"  MSER ({x},{y},{w},{h}) area={area} ratio={ratio:.2f} -> {'ENTRA' if 150 < area < 5000 and 0.90 < ratio < 5.0 and 8 < w < 70 and 12 < h < 100 else 'FILTRADO'}")

    # ---- Canny complementario ----
    edges = cv2.Canny(gray, 40, 120)
    contornos, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        ratio = h / float(w)
        if 80 < area < 5000 and 0.90 < ratio < 5.0 and 5 < w < 70 and 12 < h < 100:
            boxes.append((x, y, w, h))
            print(f"  Canny ({x},{y},{w},{h}) area={area} ratio={ratio:.2f} -> {'ENTRA' if 150 < area < 5000 and 0.90 < ratio < 5.0 and 8 < w < 70 and 12 < h < 100 else 'FILTRADO'}")

    # ---- Filtro de densidad de bordes ----
    cajas_filtradas = []
    for (x, y, w, h) in boxes:
        roi = gray[y:y+h, x:x+w]
        edges_roi = cv2.Canny(roi, 40, 120)
        densidad = np.sum(edges_roi > 0) / (w*h)
        if densidad > 0.05:
            cajas_filtradas.append((x, y, w, h))

    # ---- NMS ----
    cajas_finales = nms(cajas_filtradas, overlapThresh=0.3)

    return cajas_finales, img, gray


if __name__ == "__main__":
    img = cv2.imread("test_ocr_panels/00016_0.png")

    boxes, img, clean = detectar_caracteres(img)
    print("Caracteres detectados:", len(boxes))

    # ---- Imagen 1: umbralizada ----
    thresh = cv2.adaptiveThreshold(
        clean, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=8
    )
    umbral_vis = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(umbral_vis, (x, y), (x+w, y+h), (0, 0, 255), 1)
    cv2.imshow("Umbralizada + detecciones", umbral_vis)

    # cargar modelo
    clf = HogSvmClassifier()
    clf.load("hog_svm_model.xml")

    texto_final = ""
    lineas = agrupar_por_lineas(boxes)

    for linea in lineas:
        texto_linea = ""
        for (x, y, w, h) in linea:

            roi = clean[y:y+h, x:x+w]
            # ---- Filtro geométrico ----
            ratio = h / float(w)
            area = w * h
            elongacion = max(w, h) / float(min(w, h))

            if ratio < 0.90:
                print(f"  RATIO ({x},{y},{w},{h}) ratio={ratio:.2f}")
                continue
            if ratio > 5.0:
                print(f"  RATIO_MAX ({x},{y},{w},{h}) ratio={ratio:.2f}")
                continue
            if elongacion > 4.5:
                print(f"  ELON ({x},{y},{w},{h})")
                continue
            if area < 100 or area > 6000:
                print(f"  AREA ({x},{y},{w},{h}) area={area}")
                continue
            if w > 120 or h > 130:
                print(f"  WH ({x},{y},{w},{h})")
                continue

            # ---- Filtro de borde de imagen ----
            img_h, img_w = clean.shape[:2]
            print(f"img_h={img_h} img_w={img_w}")

            margen = 15
            margen_superior = 1
            margen_inferior = 30

            if x < margen or y < margen_superior or (x+w) > (img_w-margen) or (y+h) > (img_h-margen_inferior):
                continue
            margen_izq = 22
            margen_der = 15
            if x < margen_izq or y < margen_superior or (x+w) > (img_w-margen_der) or (y+h) > (img_h-margen_inferior):
                continue
            
            # ---- Filtro de complejidad de bordes ----
            edges_roi = cv2.Canny(roi, 40, 120)
            num_edges = np.sum(edges_roi > 0)
            if num_edges < 15:
                continue
            densidad = num_edges / float(area)
            if densidad > 0.5:
                continue
            if num_edges > 1500:
                continue

            # ---- Filtro de forma por contornos ----
            conts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(conts) == 0:
                continue
            cnt = max(conts, key=cv2.contourArea)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) > 14:
                continue
            if cv2.arcLength(cnt, True) > 500:
                continue
            # Filtro posición: flechas suelen estar muy abajo en el panel
   
            # ---- Filtro anti-símbolo (roi binarizado) ----
            roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            conts_bin, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(conts_bin) > 0:
                cnt_bin = max(conts_bin, key=cv2.contourArea)
                area_cnt_bin = cv2.contourArea(cnt_bin)
                fill_ratio_bin = area_cnt_bin / float(area)
                num_vertices_bin = len(cv2.approxPolyDP(
                    cnt_bin, 0.02 * cv2.arcLength(cnt_bin, True), True))
                print(f"  DIBUJA_CHECK ({x},{y},{w},{h}) ratio={ratio:.2f} area={area} fill={fill_ratio_bin:.2f} vert={num_vertices_bin} w={w} h={h} conts={len(conts_bin)}")

                if ratio < 1.25 and fill_ratio_bin > 0.44 and num_vertices_bin >= 8 and area > 400 and y > img_h * 0.76:
                    print(f"  ANTISIM_FLECHA ({x},{y},{w},{h})")
                    continue
                if ratio > 1.7 and fill_ratio_bin > 0.55 and area < 350:
                    print(f"  ANTISIM_PALO ({x},{y},{w},{h})")
                    continue
                if ratio < 1.15 and area > 400 and num_vertices_bin >= 9 and w > 18 and len(conts_bin) >= 2 and fill_ratio_bin > 0.38:
                    print(f"  ANTISIM_P ({x},{y},{w},{h})")
                    continue
                if area > 2000 and w > 40:
                    print(f"  ANTISIM_GRANDE ({x},{y},{w},{h})")
                    continue
                if fill_ratio_bin > 0.75 and num_vertices_bin <= 4:
                    print(f"  ANTISIM1 ({x},{y},{w},{h})")
                    continue
                if area > 1400 and (num_vertices_bin < 4 or w > 50):
                    print(f"  ANTISIM3 ({x},{y},{w},{h})")
                    continue
                if area > 800 and 0.65 < ratio < 1.45 and fill_ratio_bin < 0.55 and w < 26 and num_vertices_bin < 6:
                    print(f"  ANTISIM2 ({x},{y},{w},{h})")
                    continue
                
                if fill_ratio_bin < 0.35 and ratio < 0.95 and area > 750:
                    print(f"  ANTISIM4 ({x},{y},{w},{h})")
                    continue
                if fill_ratio_bin > 0.60 and ratio < 0.90 and area > 900:
                    print(f"  ANTISIM5 ({x},{y},{w},{h})")
                    continue
                if area > 900 and ratio > 1.8 and fill_ratio_bin < 0.50 and len(conts_bin) == 1:
                    print(f"  ANTISIM6 ({x},{y},{w},{h})")
                    continue
                # Flecha segunda: cuadrada + fill medio + y alto en imagen
                if area > 500 and 1.0 < ratio < 1.3 and fill_ratio_bin < 0.50 and y > img_h * 0.85:
                    print(f"  ANTISIM7 ({x},{y},{w},{h})")
                    continue
                # Flecha abajo izquierda: ratio < 1 + fill medio + y alto
                if ratio < 1.10 and area > 1000 and fill_ratio_bin < 0.35:
                    print(f"  ANTISIM8 ({x},{y},{w},{h})")
                    continue
                # Flecha/símbolo con fill alto y pocos vértices
                if fill_ratio_bin > 0.65 and num_vertices_bin <= 5 and ratio > 1.5:
                    print(f"  ANTISIM10 ({x},{y},{w},{h})")
                    continue
                if w < 12 and ratio > 2.0 and y > img_h * 0.85 and len(conts_bin) >= 2:
                    print(f"  ANTISIM_FINO ({x},{y},{w},{h})")
                    continue
                if area > 800 and fill_ratio_bin <= 0.51 and num_vertices_bin <= 10 and w > img_w * 0.10 and img_w > 300:
                    print(f"  ANTISIM_SIM ({x},{y},{w},{h})")
                    continue
                if area > 1000 and len(conts_bin) >= 2 and fill_ratio_bin < 0.50 and num_vertices_bin >= 8:
                    print(f"  ANTISIM_SIM2 ({x},{y},{w},{h})")
                    continue
                # ANTISIM6b - sin condición de conts
                if ratio > 1.8 and area > 500 and fill_ratio_bin < 0.45 and w < 25 and num_vertices_bin <= 4:
                    print(f"  ANTISIM6b ({x},{y},{w},{h})")
                    continue
                if area > 800 and h > img_h * 0.45 and fill_ratio_bin < 0.55 and num_vertices_bin <= 11:
                    print(f"  ANTISIM_ALTO ({x},{y},{w},{h})")
                    continue
                if ratio < 1.05 and area > 1000 and fill_ratio_bin < 0.58 and num_vertices_bin <= 9:
                    print(f"  ANTISIM_CUAD ({x},{y},{w},{h})")
                    continue
                if area > 700 and fill_ratio_bin < 0.30 and num_vertices_bin <= 9 and ratio < 1.25 and w > img_w * 0.07 and img_w > 280:
                    print(f"  ANTISIM_CUAD2 ({x},{y},{w},{h})")
                    continue
                if ratio < 1.05 and area > 1000 and fill_ratio_bin > 0.65 and num_vertices_bin <= 9:
                    print(f"  ANTISIM_CUAD3 ({x},{y},{w},{h})")
                    continue

                print(f"  DIBUJA ({x},{y},{w},{h})")
                
                print(f"PASA ({x},{y},{w},{h}) ratio={ratio:.2f} area={area} fill={fill_ratio_bin:.2f} vert={num_vertices_bin}")

            # ---- OCR ----
            roi = cv2.resize(roi, (25,25), interpolation=cv2.INTER_AREA)
            roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            roi = cv2.GaussianBlur(roi, (3,3), 0)
            roi = cv2.equalizeHist(roi)
            roi = cv2.copyMakeBorder(roi, 2,2,2,2, cv2.BORDER_CONSTANT, value=0)

            pred = clf.predict(roi)
            char = clf.label2char(pred)
            texto_linea += char

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, char, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if texto_linea:
            texto_final += texto_linea + "\n"

    print("Texto reconstruido:\n", texto_final)

    cv2.imshow("Predicciones", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
