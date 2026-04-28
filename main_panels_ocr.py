import argparse
import os
import cv2
import glob
def cargar_diccionario_imagenes(ruta_directorio):
    """
    Lee las imágenes de un directorio y las agrupa en un diccionario.
    Asume que dentro del directorio hay subcarpetas nombradas con el carácter 
    correspondiente (ej. 'A', 'b', '0') que contienen las imágenes .png.
    """
    images_dict = {}
    if not os.path.exists(ruta_directorio):
        print(f"Advertencia: No se encontró la ruta {ruta_directorio}")
        return images_dict

    for nombre_carpeta in os.listdir(ruta_directorio):
        ruta_carpeta = os.path.join(ruta_directorio, nombre_carpeta)
        
        if os.path.isdir(ruta_carpeta):
            # La clave del diccionario será el nombre de la carpeta
            caracter = nombre_carpeta 
            images_dict[caracter] = []
            
            # Buscar todas las imágenes png en esa carpeta
            patron = os.path.join(ruta_carpeta, '*.png')
            for ruta_img in glob.glob(patron):
                img = cv2.imread(ruta_img)
                if img is not None:
                    images_dict[caracter].append(img)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="LdaNormalBayes", help='Detector string name')
    parser.add_argument(
        '--train_path', default="train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    # Load training data
    print(f"Cargando datos de entrenamiento desde: {args.train_path}")
    train_dict = cargar_diccionario_imagenes(args.train_path)
   # 2. Create the OCR classifier
    print(f"Instanciando clasificador {args.detector}...")
    if args.detector == "LdaNormalBayes":
        ocr_classifier = LdaNormalBayesClassifier(ocr_char_size=(25, 25))
    else:
        # Aquí podrías instanciar otros clasificadores en el futuro (KNN, SVM...)
        ocr_classifier = LdaNormalBayesClassifier(ocr_char_size=(25, 25))

    # Entrenar el clasificador
    ocr_classifier.train(train_dict)

    # 3. Load testing data
    print(f"\nCargando datos de validación desde: {args.test_path}")
    test_dict = cargar_diccionario_imagenes(args.test_path)

    # 4. Evaluate OCR over road panels
    print("Iniciando evaluación de validación...")
    
    # Obtener etiquetas reales (Ground Truth)
    true_labels = ocr_classifier.get_labels_dict(test_dict)
    
    # Obtener etiquetas predichas por el modelo
    predicted_labels = ocr_classifier.predict_dict(test_dict)

    # Calcular y mostrar la tasa de acierto (Accuracy)
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print("\n" + "="*40)
    print("📊 RESULTADOS DE LA EVALUACIÓN OCR")
    print("="*40)
    print(f"Total caracteres testeados: {len(true_labels)}")
    print(f"Tasa de Acierto (Accuracy): {accuracy * 100:.2f}%")
    print("="*40)


