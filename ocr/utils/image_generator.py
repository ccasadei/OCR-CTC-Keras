import os
import random

import cv2
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import get_session
from tensorflow.python.lib.io import file_io

from ocr.utils import parameter


# generatore di immagini e testi per i batch
class TextImageGenerator:
    def __init__(self, dataset_dirpath, batch_size, downsample_factor, grayscale, img_wh):
        self.grayscale = grayscale
        self.downsample_factor = downsample_factor
        self.img_wh = img_wh
        self.dataset_dirpath = dataset_dirpath
        self.batch_size = batch_size

        tfrecord_fname = os.path.join(dataset_dirpath, "dataset.tfrecord")
        if file_io.file_exists(tfrecord_fname):
            self.dataset_fnames = None
            self.indexes = None
            self.cur_index = None
            self.session = get_session()
            # conto i record del tfrecord (non esiste un metodo diretto, devo scorrermi tutto il tfrecord...)
            self.n = sum(1 for _ in tf.python_io.tf_record_iterator(tfrecord_fname))
            self.next_tfrecord = tf.data.TFRecordDataset([tfrecord_fname]) \
                .map(self.extract_fn) \
                .repeat() \
                .shuffle(buffer_size=10000) \
                .make_one_shot_iterator() \
                .get_next()
        else:
            self.session = None
            self.next_tfrecord = None
            self.images_dirpath = os.path.join(dataset_dirpath, "images")
            self.annotations_dirpath = os.path.join(dataset_dirpath, "annotations")
            self.dataset_fnames = [".".join(f.split(".")[:-1]) for f in os.listdir(self.images_dirpath)]
            self.n = len(self.dataset_fnames)
            self.indexes = list(range(self.n))
            self.cur_index = self.n

    # restituisco il nome ed il contenuto del prossimo file in versione numpy.array (file immagine)
    # e stringa (annotazione), sempre in ordine alfabetico dei nomi
    def next_files(self):
        self.cur_index += 1
        # se ho superato il dataset, ricomincio da capo ordinando gli indici per la lettura informazioni
        if self.cur_index >= self.n:
            self.cur_index = 0
            self.indexes = sorted(self.indexes)

        # prendo il nome del file usando l'indice corrente
        fname = self.dataset_fnames[self.indexes[self.cur_index]]

        # leggo il contenuto del file immagine
        img = np.fromfile(os.path.join(self.images_dirpath, fname + ".jpg"), dtype=np.uint8)

        # leggo il testo associato
        file = open(os.path.join(self.annotations_dirpath, fname + ".txt"), "r")
        annotation = file.read()
        file.close()

        return fname, img, annotation

    # preparo la prossima immagine e testo associato
    def next_sample(self):
        # verifico se devo leggere da un dataset TFRecord o da file system
        if self.session is not None:
            _, imgbytes, labbytes = self.session.run(self.next_tfrecord)
            img = cv2.imdecode(np.fromstring(imgbytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            labels = np.fromstring(labbytes, dtype=np.uint8)
            textlen = labels.size
            labels = labels.tolist()
        else:
            self.cur_index += 1
            # se ho superato il dataset, ricomincio da capo randomizzando gli indici per la lettura informazioni
            if self.cur_index >= self.n:
                self.cur_index = 0
                random.shuffle(self.indexes)

            # prendo il nome del file usando l'indice corrente (che è stato randomizzato in precedenza)
            fname = self.dataset_fnames[self.indexes[self.cur_index]]
            # leggo l'immagine dal file
            img = cv2.imread(os.path.join(self.images_dirpath, fname + ".jpg"), cv2.IMREAD_UNCHANGED)

            # leggo il testo associato
            file = open(os.path.join(self.annotations_dirpath, fname + ".txt"), "r")
            annotation = file.read()
            file.close()
            labels = self.text_to_labels(annotation)
            textlen = len(annotation)

        # preparo l'immagine all'elaborazione (ridimensiono e normalizzo)
        img = self.prepara_immagine(img, self.grayscale, self.img_wh)

        # correggo la lunghezza delle labels che deve essere fissa
        labels = self.padding_labels(labels)

        return img, labels, textlen

    # creo un batch
    def next_batch(self):
        while True:
            # preparo gli array delle immagini, dei testi, delle lunghezze dell'input e delle label di ground truth
            X_data = np.ones([self.batch_size, self.img_wh[0], self.img_wh[1], 1 if self.grayscale else 3])
            Y_data = np.ones([self.batch_size, parameter.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_wh[0] // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))

            # ciclo per l'intero batch
            for i in range(self.batch_size):
                # ottengo la prossima immagine processata, testo corrente tradotto
                # in labels e lunghezza effettiva testo corrente
                img, labels, textlen = self.next_sample()
                X_data[i] = img
                # processo il testo e l'aggiungo nell'array testi
                Y_data[i] = labels
                # indico la lunghezza del testo corrente
                label_length[i] = textlen

            # preparo il dizionario con i dati da elaborare
            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length
            }

            # preparo il dizionario di output
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)

    @staticmethod
    def prepara_immagine(img, grayscale, img_wh):
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, img_wh)
        img = img.astype(np.float32)
        img = (img / 255.0) * 2.0 - 1.0
        if grayscale:
            img = img.T
            img = np.expand_dims(img, axis=-1)
        else:
            img = np.transpose(img, (1, 0, 2))
        img = np.expand_dims(img, axis=0)
        return img

    # trasforma le label in testo
    @staticmethod
    def labels_to_text(labels):
        return ''.join(list(map(lambda x: parameter.letters[int(x)], labels)))

    # trasforma il testo in label
    @staticmethod
    def text_to_labels(text):
        return list(map(lambda x: parameter.letters.index(x), text))

    # se la lunghezza non raggiunge quella massima, aggiungo una classe blank
    @staticmethod
    def padding_labels(labels):
        while len(labels) < parameter.max_text_len:
            labels.append(parameter.blank_class)
        return labels

    @staticmethod
    def extract_fn(data_record):
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'bytes': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string)
        }
        # sample = tf.parse_example([data_record], features)
        sample = tf.parse_single_example(data_record, features)
        filename = sample["filename"]
        bytes_arr = sample["bytes"]
        labels = sample["labels"]
        return filename, bytes_arr, labels
