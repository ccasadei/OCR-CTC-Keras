import os

from keras.callbacks import ModelCheckpoint

from ocr.utils import iolib, parameter


# callback di modelcheckpoint ridefinito per usare la IOLib
class ModelCheckpointIO(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super(ModelCheckpointIO, self).on_epoch_end(epoch, logs)
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        # se ho salvato i pesi, li copio in folder definitivo e cancello il file temporaneo
        with iolib.IOLib() as io:
            if io.fileExists(filepath):
                io.copy(filepath, os.path.join(parameter.weights_path, filepath))
                io.delete(filepath)
