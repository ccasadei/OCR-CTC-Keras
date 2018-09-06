# identificativi dei modelli disponibili
MODEL_LSTM = "LSTM"
MODEL_GRU = "GRU"
MODEL_GRU2 = "GRU2"
MODEL_GRU3 = "GRU3"
MODEL_GRU4 = "GRU4"
MODEL_GRU5 = "GRU5"


from keras import backend as K
from ocr.models import model_LSTM, model_GRU, model_GRU2, model_GRU3, model_GRU4, model_GRU5
from ocr.utils import parameter

K.set_learning_phase(0)


# ritorna oggetto modello (che contiene i parametri particolari di quel modello) ed il modello keras vero e proprio
def get_model_and_network(training):
    if parameter.model_type == MODEL_LSTM:
        model = model_LSTM.Model_LSTM()
    elif parameter.model_type == MODEL_GRU:
        model = model_GRU.Model_GRU()
    elif parameter.model_type == MODEL_GRU2:
        model = model_GRU2.Model_GRU2()
    elif parameter.model_type == MODEL_GRU3:
        model = model_GRU3.Model_GRU3()
    elif parameter.model_type == MODEL_GRU4:
        model = model_GRU4.Model_GRU4()
    elif parameter.model_type == MODEL_GRU5:
        model = model_GRU5.Model_GRU5()
    else:
        raise Exception("Tipo modello '" + parameter.model_type + "' non riconosciuto.")

    return model, model.get_model(training=training)
