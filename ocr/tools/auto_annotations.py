import argparse
import os
import pickle

import cv2
import numpy

from ocr.models import model_factory
from ocr.utils import parameter, iolib, image_generator


def genera_annotazioni(path_img, path_txt, null_predict_default, isgrayed, img_wh, anchor_boxes):
    # prepara l'elenco dei file immagini presenti nel path
    imflst = os.listdir(path_img)
    i = 0
    # per ogni immagine
    while i < len(imflst):
        # prepara i nomi dei file di immagine e annotazione, comprensivi di path
        imf = imflst[i]
        imgfname = os.path.join(path_img, imf)
        if os.path.isdir(imgfname):
            i += 1
            continue
        img_loaded = cv2.imread(imgfname)
        if anchor_boxes is None:
            imgs = [img_loaded]
        else:
            img_boxes_path = os.path.join(path_img, "boxes")
            if not os.path.exists(img_boxes_path):
                os.mkdir(img_boxes_path)

            imgs = []
            h, w, _ = img_loaded.shape
            for box_num, abox in enumerate(anchor_boxes):
                x1 = int(abox[0][0] * w)
                y1 = int(abox[0][1] * h)
                x2 = int(abox[1][0] * w)
                y2 = int(abox[1][1] * h)

                imgboxfname_to_save = os.path.join(img_boxes_path, str(imf.split(".")[0]) + "_{0:03d}.jpg".format(box_num))
                imgbox = img_loaded[y1:y2, x1:x2]
                cv2.imwrite(imgboxfname_to_save, imgbox)
                imgs.append(imgbox)

            i += 1
            if (i % 100) == 0:
                print("Elaborate", i * (len(anchor_boxes) if anchor_boxes is not None else 1), "immagini...")
            continue

        # nota: se arrivo qui, allora non sto elaborando gli anchor box!
        for img in imgs:
            txtfname = os.path.join(path_txt, str(imf.split(".")[0]) + ".txt")
            # prepara l'immagine da visualizzare contenente immagine e label predetta
            img2show = numpy.zeros((1024, 1024, 3), dtype=numpy.uint8)
            img_to_show = numpy.copy(img)
            img_to_show = cv2.resize(img_to_show, (img_to_show.shape[1] * 2, img_to_show.shape[0] * 2))

            img_to_show_as_input = numpy.copy(img)
            if model.isgrayscaled():
                # mando in scala di grigi e riporto in rgb perchè l'immagine "canvas" finale è rgb
                img_to_show_as_input = cv2.cvtColor(img_to_show_as_input, cv2.COLOR_BGR2GRAY)
                img_to_show_as_input = cv2.cvtColor(img_to_show_as_input, cv2.COLOR_GRAY2BGR)
            img_to_show_as_input = cv2.resize(img_to_show_as_input, (256,64))#model.get_input_image_size())

            # se esite l'annotazione su file la legge
            if os.path.exists(txtfname):
                with open(txtfname, "r") as f:
                    txt = f.read()
            # altrimenti la predice usando la rete neurale
            else:
                if null_predict_default:
                    txt = ""
                else:
                    img = image_generator.TextImageGenerator.prepara_immagine(img, isgrayed, img_wh)
                    net_out_value = model_nn.predict(img)
                    txt = model.decode_label(net_out_value)

            # prepara il testo nell'immagine da visualizzare
            ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(img2show, txt, (5, ts[0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # copia l'immagine nell'immagine da visualizzare, spostata di un offset rispetto alla scritta della annotation
            ofy = ts[0][1] + 12
            img2show[ofy:ofy + img_to_show.shape[0], :img_to_show.shape[1], :] = img_to_show

            # copia anche un'immagine dimensionata a quanto desiderato in input dalla rete neurale
            img2show[ofy + img_to_show.shape[0] + 12:  ofy + img_to_show.shape[0] + 12 + img_to_show_as_input.shape[0],
                     :img_to_show_as_input.shape[1], :] = img_to_show_as_input

            # visualizza l'immagine
            cv2.imshow("Immagine", img2show)
            # l'attesa tasti con pausa 10 ms serve a visualizzare bene l'immagine
            cv2.waitKey(10)

            # ottiene l'input da prompt
            newtxt = input('[' + str(imf.split(".")[0]) + '] - (' + str(i + 1) + '/' + str(len(imflst)) + ') - Valore attuale "' + txt + '": ')

            # gestisce i comandi ottenuti in input
            # "" = memorizza l'annotazione letta da file o predetta e va avanti
            if newtxt == "":
                with open(txtfname, "w") as f:
                    f.write(txt)
                i = i + 1
            # "a" = torna indietro
            elif newtxt == "a":
                i = i - 1
                if i < 0:
                    i = 0
            # "go xxxxxxxx" = si sposta sull'immagine con noem xxxxxxxx, se esiste
            elif newtxt.startswith("go "):
                t = 0
                while t < len(imflst):
                    if imflst[t].split(".")[0] != newtxt.split(" ")[1]:
                        t = t + 1
                    else:
                        break
                if t < len(imflst):
                    i = t
            # "q" = esce
            elif newtxt == "q":
                break
            # "x" = memorizza void
            elif newtxt == "x":
                newtxt = ""
                with open(txtfname, "w") as f:
                    f.write(newtxt)
                i = i + 1
            # altrimenti memorizza l'annotazione inserita in input e va avanti
            else:
                with open(txtfname, "w") as f:
                    f.write(newtxt)
                i = i + 1

    print("")
    print("Elaborazione completata!")
    print("Totale:", i * (len(anchor_boxes) if anchor_boxes is not None else 1), "immagini")


# prepara gli argomenti da linea di comando
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help="weight file directory",
                    type=str, default=os.path.join("../" + parameter.weights_path, parameter.model_type + '--pretrained.h5'))
parser.add_argument("-i", "--path_img", help="Images directory",
                    type=str)
parser.add_argument("-t", "--path_txt", help="Texts directory",
                    type=str)
parser.add_argument("-a", "--anchor_boxes_pickle", help="Se valorizzato, le immagini vengono ritagliate seguendo le anchor box",
                    type=str)
parser.add_argument("-n", "--null_predict_default", help="Indica se la predizione deve essere vuota di default o usare i risultati della rete neurale",
                    type=bool)
args = parser.parse_args()

# preparo il modello in base al tipo impostato nei parametri
model, model_nn = model_factory.get_model_and_network(training=False)
model_nn.summary()

print("", flush=True)

try:
    with open(args.anchor_boxes_pickle, "rb") as fp:
        anchor_boxex = pickle.load(fp)
    print("Elaborazione tramite anchor box. Estraggo solo le immagini!")
except:
    anchor_boxex = None

# se esistono, legge i pesi
if not args.null_predict_default:
    with iolib.IOLib() as io:
        if io.copyToTmp(args.weight):
            model_nn.load_weights(io.getTmpFileName(), by_name=True)
            print("...Uso i pesi già allenati...", flush=True)
        else:
            print("...NUOVO ALLENAMENTO...", flush=True)
    print("", flush=True)

# esegue la generazione delle annotazioni con supporto del modello
genera_annotazioni(args.path_img, args.path_txt, args.null_predict_default,
                   model.isgrayscaled(), model.get_input_image_size(), anchor_boxex)
