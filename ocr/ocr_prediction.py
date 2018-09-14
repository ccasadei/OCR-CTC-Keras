import argparse
import os
import pickle
import time
from random import shuffle

import cv2
import numpy

from ocr.models import model_factory
from ocr.utils import image_generator, iolib, parameter

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help="weight file directory",
                    type=str, default=os.path.join(parameter.weights_path, parameter.model_type + '--pretrained.h5'))
parser.add_argument("-t", "--test_img", help="Test image directory",
                    type=str)
parser.add_argument("-a", "--anchor_boxes_pickle", help="Se valorizzato, le immagini vengono ritagliate seguendo le anchor box ed ogni box va in predizione",
                    type=str)
args = parser.parse_args()

# preparo il modello in base al tipo impostato nei parametri
model, model_nn = model_factory.get_model_and_network(False)
model_nn.summary()

print("", flush=True)
with iolib.IOLib() as io:
    if io.copyToTmp(args.weight):
        model_nn.load_weights(io.getTmpFileName(), by_name=True)
        print("...Uso i pesi già allenati...", flush=True)
    else:
        print("...NUOVO ALLENAMENTO...", flush=True)
print("", flush=True)

test_imgs_dir = os.path.join(args.test_img, "images")
test_texts_dir = os.path.join(args.test_img, "annotations")
test_imgs = os.listdir(test_imgs_dir)
shuffle(test_imgs)

try:
    with open(args.anchor_boxes_pickle, "rb") as fp:
        anchor_boxex = pickle.load(fp)
    print("Elaborazione tramite anchor box")
except:
    anchor_boxex = None

total = 0
acc = 0
total_cifre = 0
acc_cifre = 0
start = time.time()
for nimg, test_img in enumerate(test_imgs):
    total += 1

    img_loaded = cv2.imread(os.path.join(test_imgs_dir, test_img), cv2.IMREAD_UNCHANGED)

    if anchor_boxex is None:
        imgs = [img_loaded]

        textfname = os.path.join(test_texts_dir, str(test_img.split(".")[0]) + ".txt")
        if os.path.exists(textfname):
            f = open(textfname, "r")
            text_gt = f.read()
            f.close()
        else:
            text_gt = None
    else:
        text_gt = None
        parameter.prediction_showimages = True

        imgs = []
        h, w, _ = img_loaded.shape
        for abox in anchor_boxex:
            x1 = int(abox[0][0] * w)
            y1 = int(abox[0][1] * h)
            x2 = int(abox[1][0] * w)
            y2 = int(abox[1][1] * h)

            imgs.append(img_loaded[y1:y2, x1:x2])

    for abidx, img in enumerate(imgs):
        img_pred = image_generator.TextImageGenerator.prepara_immagine(img, model.isgrayscaled(),
                                                                       model.get_input_image_size())

        img = cv2.resize(img, model.get_input_image_size())
        net_out_value = model_nn.predict(img_pred)

        pred_texts = model.decode_label(net_out_value)

        if text_gt is not None:
            for i in range(min(len(pred_texts), len(text_gt))):
                if pred_texts[i] == text_gt[i]:
                    acc_cifre += 1
            total_cifre += len(text_gt)

            if pred_texts == text_gt:
                acc += 1

        if parameter.prediction_showimages:
            if anchor_boxex is None:
                print("Immagine", test_img)
            else:
                print("Immagine", test_img, "- Anchor box nr", (abidx + 1))
            img_bkg = numpy.zeros((150, 480, 3), dtype=numpy.uint8)
            img_bkg[-img.shape[0]:, :img.shape[1]] = img
            if text_gt is not None:
                cv2.putText(img_bkg, "GT:", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img_bkg, text_gt, (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img_bkg, "PR:", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(img_bkg, pred_texts, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if text_gt is not None:
                if text_gt == pred_texts:
                    cv2.putText(img_bkg, "OK", (250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(img_bkg, "XX", (250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Immagine", img_bkg)
            if cv2.waitKey(0) == 27:
                break
            # cv2.destroyAllWindows()
        else:
            if (text_gt is not None) and ((nimg + 1) % 100 == 0):
                end = time.time()
                total_time = (end - start)
                print("Elaborate", (nimg + 1), "immagini...", flush=True)
                print("    Tempo medio elaborazione : {:.4f}".format(total_time / total), flush=True)
                print("    Accuracy media           : {:.4f}".format(acc / total), flush=True)
                print("    Accuracy media cifre     : {:.4f}".format(acc_cifre / total_cifre), flush=True)
                print("", flush=True)

end = time.time()
total_time = (end - start)
if not parameter.prediction_showimages:
    print("", flush=True)
    print("---------------------------------------------------", flush=True)
    print("   E L A B O R A Z I O N E   C O M P L E T A T A   ", flush=True)
    print("---------------------------------------------------", flush=True)
    print("", flush=True)
    print("Immagini elaborate       :", len(test_imgs), flush=True)
    if total > 0:
        print("Tempo medio elaborazione : {:.4f}".format(total_time / total), flush=True)
        print("Accuracy media           : {:.4f}".format(acc / total), flush=True)
    if total_cifre > 0:
        print("Accuracy media cifre     : {:.4f}".format(acc_cifre / total_cifre), flush=True)
