from keras import backend as K
import itertools
import numpy as np

from ocr.utils import parameter


class Model_BASE:
    # funzione loss CTC
    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        # la prima coppia di output dell'RNN tende a contenere dati sporchi, quindi li ignoro
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def get_model(self, training):
        pass

    def isgrayscaled(self):
        pass

    def get_downsample_factor(self):
        pass

    def get_input_image_size(self):
        pass

    def decode_label(self, out):
        # lo shape di out è (1, ntimes, len(chars)+1)
        # dove ntimes sono le suddivisioni temporali delle attivazioni,
        # ognuna con le probabilità di ogni carattere + blank
        # solitamente ntimes = 32, ma dipende dalla larghezza immagine e dal downsample

        # trovo il massimo indice per ognuna delle colonne temporali (lo shape di out_best sarà ntimes)
        out_best = list(np.argmax(out[0, 2:], axis=1))
        # elimino le ripetizioni di caratteri
        out_best = [k for k, g in itertools.groupby(out_best)]
        # creo la stringa decodificata scorrendo gli indici
        outstr = ''
        for i in out_best:
            # il blank ha un indice superiore alla lunghezza dell'array dei caratteri ammissibili
            # se il carattere attuale non è un blank, lo accodo alla stringa finale
            if i < len(parameter.letters):
                outstr += parameter.letters[i]
        return outstr
