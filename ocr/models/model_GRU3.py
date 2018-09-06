from keras.applications import ResNet50
from keras.layers import GRU
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers import add, concatenate
from keras.models import Model

from ocr.models.model_BASE import Model_BASE
from ocr.utils import parameter


class Model_GRU3(Model_BASE):
    # preparo il modello con GRU
    def get_model(self, training):
        img_wh = self.get_input_image_size()
        input_shape = (img_wh[0], img_wh[1], 3)

        inputs = Input(name='the_input', shape=input_shape, dtype='float32')

        m = ResNet50(include_top=False, input_tensor=inputs, input_shape=input_shape)

        inner = m.output  # (None, 32, 7, 2048)

        # passo da CNN a RNN
        inner = Reshape(target_shape=(32, 14336), name='reshape')(inner)  # (None, 32, 14336)
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

        # parte RNN
        gru_1 = GRU(512, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
        gru1_merged = add([gru_1, gru_1b])
        gru1_merged = BatchNormalization()(gru1_merged)
        gru_2 = GRU(512, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
        gru2_merged = concatenate([gru_2, gru_2b])
        gru2_merged = BatchNormalization()(gru2_merged)

        # transformo l'output RNN in attivazioni di caratteri
        inner = Dense(parameter.num_classes, kernel_initializer='he_normal', name='dense2')(gru2_merged)  # (None, 32, num_classes)
        y_pred = Activation('softmax', name='softmax')(inner)

        labels = Input(name='the_labels', shape=[parameter.max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        # per la funzione loss uso una lambda function
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        # preparo il modello a seconda se sto facendo allenamento o inferenza
        if training:
            mod = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
        else:
            mod = Model(inputs=[inputs], outputs=y_pred)

        # se devo "freezare" l'estrattore di feature convoluzionale, lo faccio
        if parameter.freeze_cnn:
            for l in mod.layers:
                if l.name == "reshape":
                    break
                l.trainable = False

        return mod

    def isgrayscaled(self):
        return False

    def get_downsample_factor(self):
        return 64

    def get_input_image_size(self):
        return (128 * 8, 200)
