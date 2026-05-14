import cv2
import numpy as np
import os

# Parámetros HSV para azul (más permisivo - incluye eléctricos)
azul_bajos = np.array([75, 40, 40], dtype=np.uint8)   # H: más abierto, S: menor, V: menor
azul_altos = np.array([180, 255, 255], dtype=np.uint8)  # H: hasta cian puro

# Parámetros para la máscara azul
alto_mascara = 50
ancho_mascara = 100
mascara_ideal = np.ones((alto_mascara, ancho_mascara), dtype=np.float32)

# Filtro de aspecto
top_ratio = 10.0   # Más permisivo (era 9.0)
bottom_ratio = 0.15  # Más permisivo (era 0.2)

# Scoring
umbral_score = 0.3   # Más bajo para detectar más (era 0.4)
umbral_iou = 0.6     # Más alto para evitar fusión (era 0.5)
resize_percentage = 0.10  # Agrandar más la caja (era 0.05)
interpolation = cv2.INTER_NEAREST

# Parámetros de filtrado de blobs
min_area_pixels = 500   # Menor para detectar más pequeños (era 1000)
max_area_pixels = 200000    # Área máxima de un blob azul


def crear_configuracion_color():
    return {
        "alto": alto_mascara,
        "ancho": ancho_mascara,
        "azul_bajos": azul_bajos,
        "azul_altos": azul_altos,
        "mascara_ideal": mascara_ideal,
        "umbral_score": umbral_score,
        "interpolation": interpolation,
    }


def cargar_imagen(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    return image


def detectar_pixeles_azules(image):
   
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mascara_azul = cv2.inRange(
        hsv_image,
        azul_bajos,
        azul_altos,
    )

    return mascara_azul


def analizar_distribucion_espacial(mascara_azul):
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mascara_azul, connectivity=8
    )

    return num_labels, labels, stats, centroids


def obtener_candidatos_desde_blobs(num_labels, stats, centroids):
    
    candidatos = []

    for i in range(1, num_labels):  # Empezar en 1, el 0 es el fondo
        x, y, w, h, area = stats[i]

        # Filtrar por área
        if area < min_area_pixels or area > max_area_pixels:
            continue

        # Filtrar por relación de aspecto
        if h == 0:
            continue
        ratio = w / float(h)
        if bottom_ratio < ratio < top_ratio:
            candidatos.append((x, y, w, h))

    return candidatos


def calcular_score_azul(recorte, config_color):
   
    if recorte.size == 0:
        return None

    recorte_resized = cv2.resize(
        recorte,
        (config_color["ancho"], config_color["alto"]),
        interpolation=config_color["interpolation"]
    )

    hsv_image = cv2.cvtColor(recorte_resized, cv2.COLOR_BGR2HSV)

    mascara_azul = cv2.inRange(
        hsv_image,
        config_color["azul_bajos"],
        config_color["azul_altos"],
    )

    mascara_normalizada = mascara_azul.astype(np.float32) / 255.0
    correlacion = np.sum(mascara_normalizada * config_color["mascara_ideal"])
    return correlacion / float(config_color["ancho"] * config_color["alto"])


def filtrar_detecciones_por_color(image, candidatos, config_color):
   
    detecciones_finales = []
    alto_imagen, ancho_imagen = image.shape[:2]

    for x, y, w, h in candidatos:
        # Agrandar ligeramente la caja
        x, y, w, h = agrandar_caja((x, y, w, h), resize_percentage)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(ancho_imagen, x + w)
        y2 = min(alto_imagen, y + h)

        recorte = image[y1:y2, x1:x2]
        score = calcular_score_azul(recorte, config_color)

        if score is None:
            continue

        if score > config_color["umbral_score"]:
            detecciones_finales.append((x1, y1, x2, y2, score))

    return detecciones_finales


def agrandar_caja(caja, porcentaje):
  
    x, y, w, h = caja
    factor = 1.0 + porcentaje
    nuevo_w = int(round(w * factor))
    nuevo_h = int(round(h * factor))
    centro_x = x + w / 2.0
    centro_y = y + h / 2.0
    nuevo_x = int(round(centro_x - nuevo_w / 2.0))
    nuevo_y = int(round(centro_y - nuevo_h / 2.0))
    return nuevo_x, nuevo_y, nuevo_w, nuevo_h


def calcular_iou(boxA, boxB):
   
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def nms_maximos_locales(detecciones, umbral_iou=umbral_iou):
  
    if not detecciones:
        return []

    detecciones = sorted(detecciones, key=lambda x: x[4], reverse=True)

    seleccionadas = []
    while len(detecciones) > 0:
        actual = detecciones.pop(0)
        seleccionadas.append(actual)

        detecciones = [
            d for d in detecciones
            if calcular_iou(actual, d) < umbral_iou
        ]

    return seleccionadas


def dibujar_detecciones_finales(image, detecciones_finales):
 
    image_copy = image.copy()

    for x1, y1, x2, y2, score in detecciones_finales:
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_copy,
            f"{score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return image_copy


def visualizar_blobs(image, mascara_azul, num_labels, stats, centroids):
  
    # Crear imagen RGB para mostrar los blobs
    h, w = mascara_azul.shape
    blobs_viz = np.zeros((h, w, 3), dtype=np.uint8)

    # Colores predefinidos para los primeros blobs
    colores = [
        (255, 0, 0),    # Rojo
        (0, 255, 0),    # Verde
        (0, 0, 255),    # Azul
        (255, 255, 0),  # Cian
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Amarillo
    ]

    for i in range(1, min(num_labels, len(colores) + 1)):
        mascara_blob = (mascara_azul == i).astype(np.uint8) * 255
        blobs_viz[mascara_blob > 0] = colores[i % len(colores)]

        # Dibujar centroide
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        cv2.circle(blobs_viz, (cx, cy), 5, (255, 255, 255), -1)

        # Dibujar bbox
        x, y, w, h = stats[i][:4]
        cv2.rectangle(blobs_viz, (x, y), (x + w, y + h), (255, 255, 255), 1)

    return blobs_viz


def detectar_color_distribucion(test_path):
   
    archivo_resultados = open("resultado.txt", "w")

    config_color = crear_configuracion_color()

    print(f"Procesando imágenes en: {test_path} con Color+Distribución Espacial...")

    for img_path in test_path.glob("*.png"):
        nombre_fichero = img_path.name
        img = cargar_imagen(img_path)

        # Paso 1: Detectar píxeles azules (máscara HSV)
        mascara_azul = detectar_pixeles_azules(img)

        # Paso 2: Analizar distribución espacial con connectedComponents
        num_labels, labels, stats, centroids = analizar_distribucion_espacial(mascara_azul)

        print(f"  {nombre_fichero}: {num_labels - 1} blobs azules detectados")

        # Paso 3: Obtener candidatos desde los blobs
        candidatos = obtener_candidatos_desde_blobs(num_labels, stats, centroids)

        print(f"  Candidatos tras filtro aspecto: {len(candidatos)}")

        # Paso 4: Filtrar por score de color
        detecciones = filtrar_detecciones_por_color(img, candidatos, config_color)

        # Paso 5: NMS
        detecciones_finales = nms_maximos_locales(detecciones)

        # Guardar resultados
        for x1, y1, x2, y2, score in detecciones_finales:
            linea = f"{nombre_fichero};{x1};{y1};{x2};{y2};1;{score:.4f}\n"
            archivo_resultados.write(linea)

        # Guardar imagen con detecciones
        img_final = dibujar_detecciones_finales(img, detecciones_finales)
        ruta_guardado = os.path.join("resultado_imgs", f"cd_{nombre_fichero}")
        cv2.imwrite(ruta_guardado, img_final)

        # Guardar máscara de azules para debug
        cv2.imwrite(os.path.join("resultado_imgs", f"cd_mask_{nombre_fichero}"), mascara_azul)

        print(f"  -> {len(detecciones_finales)} detecciones finales")

    archivo_resultados.close()
    print("Proceso finalizado. Resultados en 'resultado_color_distribucion.txt'")
