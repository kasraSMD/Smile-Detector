import cv2
import numpy as np
from tqdm import tqdm
import shutil
import os


class Preprocess:
    def __init__(self, data_dir, labels_file_path, cascade_classifier_file_path, cropped_pics_dir, Camera=False):
        if not Camera:
            self.data_dir = data_dir
            self.labels_file_path = labels_file_path
            self.cascade_classifier_file_path = cascade_classifier_file_path
            self.cropped_pics_dir = cropped_pics_dir
            if os.path.exists(cropped_pics_dir):
                try:
                    shutil.rmtree(cropped_pics_dir)
                except OSError as e:
                    print(f"Error deleting folder: {e}")
            os.makedirs(cropped_pics_dir)

            self.data, self.labels, self.base_paths = self.preprocess()
            self.save_cropped_images(self.data, self.base_paths)

        else:
            self.cascade_classifier_file_path = cascade_classifier_file_path

    def preprocess(self):
        data, labels, base_paths = self.load_dataset(self.data_dir, self.labels_file_path)
        for i, image in enumerate(tqdm(data)):
            crop_data = self.crop_face(image, self.cascade_classifier_file_path)
            if crop_data:
                data[i] = crop_data[0]
        for i, image in enumerate(tqdm(data)):
            data[i] = cv2.resize(image, (256, 256))
        return data, labels, base_paths

    def load_dataset(self, data_dir, labels_file_path):
        data = []
        labels = []
        base_paths = []
        with open(labels_file_path) as f:
            for line in tqdm(f.readlines(), desc="load labels"):
                label = line.split()[0]
                labels.append(int(label))
        for filename in tqdm(os.listdir(data_dir), desc="load images"):
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path)
            data.append(img)
            base_paths.append(filename)

        return data, labels, base_paths

    def crop_face(self, image, cascade_classifier_file_path, multi=False):
        face_cascade = cv2.CascadeClassifier(cascade_classifier_file_path)
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces):
            if multi:
                cropped_faces = []
                for (x, y, w, h) in faces:
                    face_image = image[y:y + h, x:x + w]
                    cropped_faces.append((face_image, [x, y, w, h]))
                return cropped_faces
            (x, y, w, h) = faces[0]
            return image[y:y + h, x:x + w], [x, y, w, h]
        return None

    def save_cropped_images(self, data, base_paths):
        for i, image in enumerate(tqdm(data)):
            cv2.imwrite(fr'{self.cropped_pics_dir}\{base_paths[i]}', image)

    def load_cropped_images(self):
        data = []
        for filename in tqdm(os.listdir(f'{self.cropped_pics_dir}'),
                             desc="load images"):
            img = cv2.imread(fr'{self.cropped_pics_dir}/{filename}')
            data.append(img)
        return data
