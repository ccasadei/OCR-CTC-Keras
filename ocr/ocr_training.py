import os

from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adadelta, SGD, Adam

from ocr.models import model_factory
from ocr.utils import iolib, parameter
from ocr.utils.checkpoint import ModelCheckpointIO
from ocr.utils.image_generator import TextImageGenerator


# preparo il modello in base al tipo impostato nei parametri
model, model_nn = model_factory.get_model_and_network(True)
model_nn.summary()

with iolib.IOLib() as io:
    wfname = os.path.join(parameter.weights_path, parameter.model_type + "--pretrained.h5")
    print("Cerco pesi di preallenamento in '" + wfname + "'...", flush=True)
    if io.copyToTmp(wfname):
        print("...Trovati...Li carico...", flush=True)
        model_nn.load_weights(io.getTmpFileName(), by_name=True, skip_mismatch=True)
        print("...Pesi caricati!", flush=True)
    else:
        print("...Nessun peso trovato...Procedo con un nuovo allenamento!", flush=True)

# preparo i generatori di train e valutazione
tigen_train = TextImageGenerator(parameter.train_file_path, parameter.train_batch_size,
                                 model.get_downsample_factor(), model.isgrayscaled(),
                                 model.get_input_image_size())

print("Dataset di training:", tigen_train.n, "samples", flush=True)

tigen_val = TextImageGenerator(parameter.val_file_path, parameter.val_batch_size,
                               model.get_downsample_factor(), model.isgrayscaled(),
                               model.get_input_image_size())

print("Dataset di validation:", tigen_val.n, "samples", flush=True)

# preparo i callback
callbacks = [
    # ReduceLROnPlateau(factor=0.5, patience=15, min_lr=1e-6, verbose=1),
    # EarlyStopping(min_delta=0.0001, patience=80, verbose=1),
    ModelCheckpointIO(filepath=parameter.model_type + '--{val_loss:.3f}.h5',
                      save_best_only=True, save_weights_only=True,
                      verbose=1)
]

# il calcolo della funzione loss avviene altrove, per compilare uso quindi una funzione lambda fittizia
model_nn.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr=parameter.lr))  # Adadelta(lr=parameter.lr))

# allena il modello
model_nn.fit_generator(generator=tigen_train.next_batch(),
                       steps_per_epoch=int(tigen_train.n / parameter.train_batch_size),
                       epochs=parameter.epochs,
                       callbacks=callbacks,
                       validation_data=tigen_val.next_batch(),
                       validation_steps=int(tigen_val.n / parameter.val_batch_size))

print("", flush=True)
print("ALLENAMENTO COMPLETATO!", flush=True)
print("", flush=True)
