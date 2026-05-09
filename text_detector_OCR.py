import cv2
import numpy as np
from hog_svm_classifier import HogSvmClassifier


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
    idxs = np.argsort(y2)

    pick = []

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[idxs[:-1]]

        idxs = np.delete(idxs, np.concatenate(([len(idxs)-1],
            np.where(overlap > overlapThresh)[0])))

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
        if 80 < area < 5000 and 0.6 < ratio < 5.0 and 8 < w < 70 and 10 < h < 100:
            boxes.append((x, y, w, h))
            print(f"  MSER ({x},{y},{w},{h}) area={area} ratio={ratio:.2f} -> {'ENTRA' if 80 < area < 5000 and 0.6 < ratio < 5.0 and 8 < w < 70 and 10 < h < 100 else 'FILTRADO'}")

    # ---- Canny complementario ----
    # ---- Canny complementario ----
    # ---- Canny complementario ----
    edges = cv2.Canny(gray, 40, 120)
    # Sin dilatar - usar edges directamente
    contornos, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        ratio = h / float(w)
        if 80 < area < 5000 and 0.6 < ratio < 5.0 and 8 < w < 70 and 10 < h < 100:
            boxes.append((x, y, w, h))
            print(f"  Canny ({x},{y},{w},{h}) area={area} ratio={ratio:.2f} -> {'ENTRA' if 80 < area < 5000 and 0.6 < ratio < 5.0 and 8 < w < 70 and 10 < h < 100 else 'FILTRADO'}")

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
    img = cv2.imread("test_ocr_panels/00003_0.png")

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

            # ---- Filtro de borde de imagen ----
            img_h, img_w = clean.shape[:2]
            margen = 15
            if x < margen or y < margen or (x+w) > (img_w-margen) or (y+h) > (img_h-margen):
                continue

            # ---- Filtro geométrico ----
            ratio = h / float(w)
            area = w * h
            elongacion = max(w, h) / float(min(w, h))

            if ratio < 0.4 or ratio > 5.0:
                continue
            if elongacion > 4.5:
                continue
            if area < 150 or area > 6000:
                continue
            if w > 120 or h > 130:
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

            # ---- Filtro anti-símbolo (roi binarizado) ----
            # ---- Filtro anti-símbolo (roi binarizado) ----
            roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            conts_bin, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(conts_bin) > 0:
                cnt_bin = max(conts_bin, key=cv2.contourArea)
                area_cnt_bin = cv2.contourArea(cnt_bin)
                fill_ratio_bin = area_cnt_bin / float(area)
                num_vertices_bin = len(cv2.approxPolyDP(
                    cnt_bin, 0.02 * cv2.arcLength(cnt_bin, True), True))

                # Solo rechaza símbolos MUY obvios
                if fill_ratio_bin > 0.75 and num_vertices_bin <= 4:
                    continue
                if area > 2500:
                    continue
                if area > 1200 and 0.75 < ratio < 1.3 and fill_ratio_bin < 0.50:
                    continue
                # Flecha abajo: cuadrada-ancha + fill bajo + area media
                if fill_ratio_bin < 0.35 and ratio < 0.95 and area > 750:
                    continue
                # P de parking: ancha + fill alto + area grande
                if fill_ratio_bin > 0.60 and ratio < 0.90 and area > 900:
                    continue
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
