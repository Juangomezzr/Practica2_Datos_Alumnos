from fileinput import filename
from pyexpat import features

import cv2
import numpy as np
from skimage.feature import hog
from ocr_classifier import OCRClassifier

class HogSvmClassifier(OCRClassifier):

    def __init__(self, ocr_char_size=(25,25)):
        super().__init__(ocr_char_size)
        self.svm = cv2.ml.SVM_create()
        self.svm.setKernel(cv2.ml.SVM_RBF)
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.classifier_name = "hog_svm"

    def _hog_features(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        img = cv2.resize(img, self.ocr_char_size, interpolation=cv2.INTER_AREA)
    
    # Mismo preprocesado que train_hog_svm.py
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.equalizeHist(img)
        img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    # resize final a 25x25 porque el borde lo dejó en 29x29
        img = cv2.resize(img, self.ocr_char_size)

        features = hog(
        img,
        orientations=9,
        pixels_per_cell=(5, 5),
        cells_per_block=(1, 1),
        visualize=False
        )
        return features


    def train(self, images_dict):
        C = []
        E = []

        for key in sorted(images_dict.keys()):
            for img in images_dict[key]:
                feat = self._hog_features(img)
                C.append(feat)
                E.append(self.char2label(key))

        C = np.array(C, dtype=np.float32)
        E = np.array(E, dtype=np.int32)

        self.svm.train(C, cv2.ml.ROW_SAMPLE, E)

    def predict(self, img):
        feat = self._hog_features(img).reshape(1, -1).astype(np.float32)
        ret, pred = self.svm.predict(feat)
        return int(pred[0][0])
    def save(self, filename):
        self.svm.save(filename)

    def load(self, filename):
        self.svm = cv2.ml.SVM_load(filename)
