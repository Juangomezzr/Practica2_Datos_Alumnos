import cv2
import numpy as np
import os

delta = 1 # Mantener
min_area = 5000
max_area = 200000
max_variation=0.01
top_ratio = 9.0 # Mantener
bottom_ratio = 0.2 # Mantener
resize_percentage = 0.05
alto = 50
ancho = 100
azul_bajos = np.array([90, 50, 40], dtype=np.uint8) # Mantener
azul_altos = np.array([150, 255, 255], dtype=np.uint8) # Mantener
mascara_ideal = np.ones((alto, ancho), dtype=np.float32)
umbral_score = 0.6
umbral_iou = 0.5
interpolation = cv2.INTER_NEAREST
    
def crear_detector_mser():
    return cv2.MSER_create(
        delta,
        min_area,
        max_area,
        max_variation,
    )


def cargar_imagen(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    return image


# Equializacion por tiles
def preprocesar_imagen(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(10,10))
    equalized_image = clahe.apply(gray_blur)
    return gray_image, equalized_image

# Filatrdoo de relacion aspecto
def obtener_candidatos(detector, equalized_image):
    _, bboxes = detector.detectRegions(equalized_image)

    candidatos = []
    for x, y, w, h in bboxes:
        ratio = w / float(h)
        if bottom_ratio < ratio < top_ratio:
            candidatos.append((x, y, w, h))

    return bboxes, candidatos


# Agrandar cuadro manteniendo el centro.
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

# Configuracion filtro de color
def crear_configuracion_color():   

    return {
        "alto": alto,
        "ancho": ancho,
        "azul_bajos": azul_bajos,
        "azul_altos": azul_altos,
        "mascara_ideal": mascara_ideal,
        "umbral_score": umbral_score,
    }

# <------ Filtro de color -------->
def calcular_score_azul(recorte, config_color):
    if recorte.size == 0:
        return None

    recorte_resized = cv2.resize(
        recorte,
        (config_color["ancho"], config_color["alto"]),
        interpolation=interpolation
    )

    hsv_image = cv2.cvtColor(recorte_resized, cv2.COLOR_BGR2HSV)

    #Mascara binaria de los azules si el pixel esta fuera del rango es 0
    mascara_azul = cv2.inRange(
        hsv_image,
        config_color["azul_bajos"],
        config_color["azul_altos"],
    )
    # Pasamos a 0.0 - 1.0 para calcular mas facilmente el calculo posterior
    mascara_normalizada = mascara_azul.astype(np.float32) / 255.0
    correlacion = np.sum(mascara_normalizada * config_color["mascara_ideal"])
    return correlacion / float(config_color["ancho"] * config_color["alto"])


def filtrar_detecciones_por_color(image, candidatos, config_color):
    detecciones_finales = []
    
    alto_imagen, ancho_imagen = image.shape[:2]

    for x, y, w, h in candidatos:
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

def calcular_iou(boxA, boxB):
    # box = (x1, y1, x2, y2, score)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Área de la intersección
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Áreas de cada caja
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU = Intersección / (Área A + Área B - Intersección)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def nms_maximos_locales(detecciones, umbral_iou=umbral_iou):
    if not detecciones:
        return []

    # 1. Ordenar por SCORE de mayor a menor (priorizamos confianza sobre tamaño)
    # El PDF sugiere elegir la ventana con mayor score 
    detecciones = sorted(detecciones, key=lambda x: x[4], reverse=True)
    
    seleccionadas = []
    while len(detecciones) > 0:
        actual = detecciones.pop(0)
        seleccionadas.append(actual)
        
        # 2. Filtrar el resto: eliminar las que solapen mucho con la 'actual'
        # porque consideramos que son el mismo panel detectado varias veces 
        detecciones = [
            d for d in detecciones 
            if calcular_iou(actual, d) < umbral_iou
        ]
        
    return seleccionadas

def dibujar_candidatos(image, candidatos):
    image_copy = image.copy()
    for x, y, w, h in candidatos:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image_copy


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

def detectar_mser(test_path):
     # Abrimos el archivo de texto en modo escritura ("w")
    archivo_resultados = open("resultado.txt", "w")

    # Paso 2: crear detector y configuración
    mser = crear_detector_mser()
    config_color = crear_configuracion_color()

    print(f"Procesando imágenes en: {test_path}...")

    # Iteramos sobre todos los archivos .png en la carpeta de test
    for img_path in test_path.glob("*.png"):
        nombre_fichero = img_path.name
        img = cargar_imagen(img_path)

        # Paso 3: preprocesar para MSER.
        gray, eq = preprocesar_imagen(img)

        # Paso 4: detectar regiones y filtrar candidatos geométricos.

        bboxes, candidatos = obtener_candidatos(mser, eq) 




        # Paso 5 y 6: calcular score por correlación y eliminar solapadas.
        detecciones = filtrar_detecciones_por_color(img, candidatos, config_color)
        detecciones_finales = nms_maximos_locales(detecciones)

        # Paso 7: Escribir en el archivo de texto y guardar imagen
        for x1, y1, x2, y2, score in detecciones_finales:
            # Formato: <nombre_fichero>;<x1>;<y1>;<x2>;<y2>;<tipo>;<score>
            # El tipo siempre es 1.
            linea = f"{nombre_fichero};{x1};{y1};{x2};{y2};1;{score:.4f}\n"
            archivo_resultados.write(linea)

        # Dibujar rectángulos y guardar la imagen en resultado_imgs
        img_final = dibujar_detecciones_finales(img, detecciones_finales)
        ruta_guardado = os.path.join("resultado_imgs", nombre_fichero)
        cv2.imwrite(ruta_guardado, img_final)
        
        print(f"Procesada {nombre_fichero} -> {len(detecciones_finales)} detecciones")

    # Cerramos el archivo al terminar
    archivo_resultados.close()
    print("Proceso finalizado. Resultados guardados en 'resultado.txt' y 'resultado_imgs/'")
