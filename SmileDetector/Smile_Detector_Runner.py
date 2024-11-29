import time
from threading import Thread
from Preprocess import Preprocess
import cv2
from keras.applications.inception_resnet_v2 import preprocess_input
import mediapipe as mp
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np


class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


def detect_smile(img, model, threshold=0.5):
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
    return pred[0][1] >= threshold


def preprocess_image(face):
    face = cv2.resize(face, (256, 256))
    face = preprocess_input(face)
    return face


def load_camera(model, preprocessor):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    try:
        while True:

            frame = vs.read()
            f = preprocessor.crop_face(frame, haarcascade_frontalface_default_file)
            if f:
                face = preprocess_image(f[0])
                smile_detected = detect_smile(face, model)

                cv2.rectangle(frame, (f[1][0], f[1][1]), (f[1][0] + f[1][2], f[1][1] + f[1][3]), (222, 232, 30), 2)

                if smile_detected:
                    cv2.putText(frame, f"Smile {smile_detected}", (f[1][0], f[1][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Smile {smile_detected}", (f[1][0], f[1][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    train_pictures_path = r".\GENKI-4K Face, Expression, and Pose Dataset\files"
    cropped_pictures_dir = r".\cropped_pictures"
    labels_file_path = r".\GENKI-4K Face, Expression, and Pose Dataset\labels.txt"
    haarcascade_frontalface_default_file = "haarcascade_frontalface_default.xml"

    model = keras.saving.load_model("Smile_Detector_Model_InceptionResNetV2.keras")
    load_camera(model, Preprocess(data_dir=train_pictures_path, labels_file_path=labels_file_path,
                                  cascade_classifier_file_path=haarcascade_frontalface_default_file,
                                  cropped_pics_dir=cropped_pictures_dir, Camera=True))
