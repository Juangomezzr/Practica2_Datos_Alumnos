import cv2
import numpy as np
from sklearn.decomposition import PCA
from ocr_classifier import OCRClassifier

class PcaBayesClassifier(OCRClassifier):

    def __init__(self, ocr_char_size=(25,25), n_components=50):
        super().__init__(ocr_char_size)
        self.pca = PCA(n_components=n_components)
        self.bayes = cv2.ml.NormalBayesClassifier_create()
        self.classifier_name = "pca_bayes"

    def train(self, images_dict):
        C = []
        E = []

        for key in sorted(images_dict.keys()):
            for img in images_dict[key]:
                feat = self._extraer_caracteristicas(img)
                C.append(feat)
                E.append(self.char2label(key))

        C = np.array(C, dtype=np.float32)
        E = np.array(E, dtype=np.int32)

        # PCA
        C_reduced = self.pca.fit_transform(C).astype(np.float32)

        # Entrenar Bayes
        self.bayes.train(C_reduced, cv2.ml.ROW_SAMPLE, E)

    def predict(self, img):
        feat = self._extraer_caracteristicas(img).reshape(1, -1)
        feat_pca = self.pca.transform(feat).astype(np.float32)

        ret, pred = self.bayes.predict(feat_pca)
        return int(pred[0][0])
