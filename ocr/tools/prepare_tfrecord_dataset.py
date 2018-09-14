import os

import numpy
import tensorflow as tf

from ocr.utils import image_generator, parameter


# gestione delle feature come lista bytes
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


# prepara il tfrecord
def prepara_dataset(tipo, path_src, batch_size):
    tfrecord_fname = os.path.join(path_src, "dataset.tfrecord")
    if os.path.exists(tfrecord_fname):
        os.remove(tfrecord_fname)

    # prepara il text generator per leggere velocemente il dataset
    gen = image_generator.TextImageGenerator(path_src, batch_size=batch_size, downsample_factor=None, grayscale=None, img_wh=None)

    # prepara il file tfrecord
    writer = tf.python_io.TFRecordWriter(tfrecord_fname)

    print("Dataset di", tipo, ":", gen.n, "immagini da elaborare", flush=True)
    # per ogni immagine letta dall'image generator...
    for i in range(gen.n):
        # legge nome file, immagine bytes, testo annotazione
        fname, img, text = gen.next_files()
        # trasforma l'annotazione da stringa a labels
        labels = image_generator.TextImageGenerator.text_to_labels(text)

        # prepara il record corrente
        record = tf.train.Example(features=tf.train.Features(feature={
            'filename': _bytes_feature([tf.compat.as_bytes(fname)]),
            'bytes': _bytes_feature([tf.compat.as_bytes(img.tobytes())]),
            'labels': _bytes_feature([tf.compat.as_bytes(numpy.array(labels, dtype=numpy.uint8).tobytes())])
        }))

        # accorda il record al file tfrecord
        writer.write(record.SerializeToString())

        # logga l'avanzamento
        if (i + 1) % 1000 == 0:
            print("...elaborate", (i + 1), "immagini", flush=True)
    # chiude il tfrecord ed esce
    writer.close()
    print("Dataset di", tipo, "COMPLETATO!", flush=True)
    print("", flush=True)


# prepara i dataset di train e di validazione
prepara_dataset("train", '../'+parameter.train_file_path, parameter.train_batch_size)
prepara_dataset("val", '../' + parameter.val_file_path, parameter.val_batch_size)
