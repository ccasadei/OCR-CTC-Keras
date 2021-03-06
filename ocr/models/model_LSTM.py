from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import LSTM
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers import add, concatenate
from keras.models import Model

from ocr.models.model_BASE import Model_BASE
from ocr.utils import parameter


class Model_LSTM(Model_BASE):
    # preparo il modello con LSTM (monocromatico)
    def get_model(self, training):
        img_wh = self.get_input_image_size()
        input_shape = (img_wh[0], img_wh[1], 1)  # (128, 64, 1)

        inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

        # estrattore convoluzionale (VGG)
        inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  # (None, 128, 64, 64)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

        inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

        inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

        inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

        inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)

        # passo da CNN a RNN
        inner = Reshape(target_shape=(32, 2048), name='reshape')(inner)  # (None, 32, 2048)
        inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

        # parte RNN
        lstm_1 = LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
        lstm_1b = LSTM(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
        lstm1_merged = add([lstm_1, lstm_1b])
        lstm1_merged = BatchNormalization()(lstm1_merged)
        lstm_2 = LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
        lstm_2b = LSTM(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
        lstm2_merged = concatenate([lstm_2, lstm_2b])
        lstm2_merged = BatchNormalization()(lstm2_merged)

        # transformo l'output RNN in attivazioni di caratteri
        inner = Dense(parameter.num_classes, kernel_initializer='he_normal', name='dense2')(lstm2_merged)  # (None, 32, num_classes)
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
        return True

    def get_downsample_factor(self):
        return 4

    def get_input_image_size(self):
        return (128, 64)
