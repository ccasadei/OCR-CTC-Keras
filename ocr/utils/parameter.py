from ocr.models.model_factory import MODEL_GRU, MODEL_GRU2, MODEL_LSTM, MODEL_GRU3, MODEL_GRU4, MODEL_GRU5, MODEL_GRU6

# parametri di ambiente running

run_on_cloud = True

# parametri prediction

prediction_showimages = False

# parametri del modello

model_type = MODEL_GRU6
max_text_len = 11

# parametri OCR

CHAR_VECTOR = "0123456789/."  # "0123456789/-. "   TODO: text_to_labels
letters = [letter for letter in CHAR_VECTOR]
num_classes = len(letters) + 1  # aggiungo 1 classe per il "blank"
blank_class = len(letters)

# parametri di training

epochs = 1000 if run_on_cloud else 10000
train_batch_size = 16
val_batch_size = 8
lr = 0.0001
freeze_cnn = False

# parametri folders

weights_path = "gs://casadei-cristiano/OCR/weights/" if run_on_cloud else "../weights/"
train_file_path = 'gs://casadei-cristiano/OCR/dataset/train/' if run_on_cloud else '../../Dataset/OCR.classico/train'  # '../../Dataset_AutoGen/train/'
val_file_path = 'gs://casadei-cristiano/OCR/dataset/val/' if run_on_cloud else '../../Dataset/OCR.classico/val'  # '../../Dataset_AutoGen/val/'


