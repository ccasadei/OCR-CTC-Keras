from ocr.models.model_factory import MODEL_GRU, MODEL_GRU2, MODEL_LSTM, MODEL_GRU3, MODEL_GRU4, MODEL_GRU5

# parametri di ambiente running

run_on_cloud = False

# parametri prediction

prediction_showimages = False

# parametri del modello

model_type = MODEL_GRU
max_text_len = 11

# parametri OCR

CHAR_VECTOR = "0123456789/-. "
letters = [letter for letter in CHAR_VECTOR]
num_classes = len(letters) + 1  # aggiungo 1 classe per il "blank"
blank_class = len(letters)

# parametri di training

epochs = 1000 if run_on_cloud else 10000
train_batch_size = 32
val_batch_size = 16
lr = 0.001
freeze_cnn = False

# parametri folders

weights_path = "gs://ccasadei-test/ocr/weights/" if run_on_cloud else "../weights/"
train_file_path = 'gs://ccasadei-test/ocr/Dataset/train/' if run_on_cloud else '../../Dataset/OCR/train'  # '../../Dataset_AutoGen/train/'
val_file_path = 'gs://ccasadei-test/ocr/Dataset/val/' if run_on_cloud else '../../Dataset/OCR/val'  # '../../Dataset_AutoGen/val/'
