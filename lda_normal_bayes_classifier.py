# @brief LdaNormalBayesClassifier
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025

# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminar la implementación
# Modificar como se crea conveniente (incluyendo métodos y parámetros), únicamente es una guía.

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .ocr_classifier import OCRClassifier

class LdaNormalBayesClassifier(OCRClassifier):
    """
    Classifier for Optical Character Recognition using LDA and the Bayes with Gaussian classfier.
    """

    def __init__(self, ocr_char_size):
        super().__init__(ocr_char_size)
        self.lda = LinearDiscriminantAnalysis()
        self.classifier = cv2.ml.NormalBayesClassifier_create()

    def _extraer_caracteristicas(self, img):
        """
        Aplica los pasos requeridos: escala de grises, adaptiveThreshold, 
        findContours, boundingRect, resize a 25x25 y aplanado.
        """
        # 1. Asegurar escala de grises
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 2. Umbralización adaptativa (las letras son claras sobre fondo oscuro o viceversa)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # 3. Buscar el contorno del carácter
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            # Asumimos que el contorno más grande es la letra
            c = max(contornos, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            recorte = gray[y:y+h, x:x+w]
        else:
            # Si falla, usamos la imagen original
            recorte = gray

        # 4. Redimensionar a tamaño fijo (25x25)
        resized = cv2.resize(recorte, self.ocr_char_size)

        # 5. Aplanar a un vector de 1 dimensión (tamaño 625)
        vector_caracteristicas = resized.flatten()
        return vector_caracteristicas

    def train(self, images_dict):
        """.
        Given character images in a dictionary of list of char images of fixed size, 
        train the OCR classifier. The dictionary keys are the class of the list of images 
        (or corresponding char).

        :images_dict is a dictionary of images (name of the images is the key)
        """

        # Take training images and do feature extraction
        
        X = ... # Feature vectors by rows
        y = ... # Labels for each row in X 

        # 1. Construir las matrices C (características) y E (etiquetas)
        for char_key, list_images in images_dict.items():
            label = self.char2label(char_key)
            for img in list_images:
                features = self._extraer_caracteristicas(img)
                X.append(features)
                y.append(label)
        # Scikit-learn acepta float64 (por defecto en np), pero preparamos el array
        samples = np.array(X, dtype=np.float64) 
        labels = np.array(y, dtype=np.int32)        


        # Perform LDA training

        print("Entrenando proyector LDA...")
        # 2. Perform LDA training and dimension reduction (Generar Matriz CR)
        X_reduced = self.lda.fit_transform(samples, labels)

        # Perform Classifier training

        print("Entrenando clasificador Bayesiano Normal...")
        # 3. Perform Classifier training (OpenCV ml)
        # REGLA DE ORO DE OPENCV: Las características deben ser obligatoriamente np.float32
        X_reduced_cv = np.array(X_reduced, dtype=np.float32)
        
        self.classifier.train(X_reduced_cv, cv2.ml.ROW_SAMPLE, labels)

        return samples, labels

    def predict(self, img):
        """.
        Given a single image of a character already cropped classify it.

        :img Image to classify
        
        """
        
        # 1. Extraer características (escala de grises, umbral, recorte, resize a 25x25 y aplanado)
        # Usamos la función auxiliar que te pasé en el mensaje anterior
        features = self._extraer_caracteristicas(img)
        
        # 2. Scikit-learn (LDA) espera una matriz 2D (varias filas), 
        # así que metemos nuestra única imagen dentro de unos corchetes [ ]
        features_matriz = np.array([features], dtype=np.float64)

        # 3. Reducir la dimensión usando el modelo LDA que ya entrenamos antes
        features_reduced = self.lda.transform(features_matriz)

        # 4. Convertir a float32 OBLIGATORIAMENTE para que el Bayes de OpenCV lo acepte
        features_reduced_cv = np.array(features_reduced, dtype=np.float32)

        # 5. AQUÍ ESTÁ TU "y = ..."
        # Le pasamos la característica reducida al clasificador entrenado
        _, results = self.classifier.predict(features_reduced_cv)

        # OpenCV devuelve los resultados dentro de una matriz 2D (ej: [[3.0]])
        # Extraemos el valor para devolverlo como un entero (ej: 3)
        y = results[0][0]

        return int(y)



