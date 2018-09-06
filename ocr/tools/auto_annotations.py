import argparse
import os
import cv2
import numpy

from ocr.models import model_factory
from ocr.utils import parameter, iolib, image_generator


def genera_annotazioni(path_img, path_txt, isgrayed, img_wh):
    # prepara l'elenco dei file immagini presenti nel path
    imflst = os.listdir(path_img)
    i = 0
    # per ogni immagine
    while i < len(imflst):
        # prepara i nomi dei file di immagine e annotazione, comprensivi di path
        imf = imflst[i]
        imgfname = os.path.join(path_img, imf)
        txtfname = os.path.join(path_txt, str(imf.split(".")[0]) + ".txt")

        # prepara l'immagine da visualizzare contenente immagine e label predetta
        img2show = numpy.zeros((1024, 1024, 3), dtype=numpy.uint8)
        img = cv2.imread(imgfname)
        img_to_show = numpy.copy(img)
        img_to_show = cv2.resize(img_to_show, (img_to_show.shape[1] * 2, img_to_show.shape[0] * 2))

        # se esite l'annotazione su file la legge
        if os.path.exists(txtfname):
            with open(txtfname, "r") as f:
                txt = f.read()
        # altrimenti la predice usando la rete neurale
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

        # visualizza l'immagine
        cv2.imshow("Immagine", img2show)
        # l'attesa tasti con pausa 10 ms serve a visualizzare bene l'immagine
        cv2.waitKey(10)

        # ottiene l'input da prompt
        newtxt = input('[' + str(imf.split(".")[0]) + '] - ('+str(i+1)+'/'+str(len(imflst))+') - Valore attuale "' + txt + '": ')
        cv2.destroyAllWindows()

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
        # altrimenti memorizza l'annotazione inserita in input e va avanti
        else:
            with open(txtfname, "w") as f:
                f.write(newtxt)
            i = i + 1


# prepara gli argomenti da linea di comando
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help="weight file directory",
                    type=str, default=os.path.join("../"+parameter.weights_path, parameter.model_type + '--pretrained.h5'))
parser.add_argument("-i", "--path_img", help="Images directory",
                    type=str)
parser.add_argument("-t", "--path_txt", help="Texts directory",
                    type=str)
args = parser.parse_args()

# preparo il modello in base al tipo impostato nei parametri
model, model_nn = model_factory.get_model_and_network(training=False)
model_nn.summary()

print("", flush=True)
# se esistono, legge i pesi
with iolib.IOLib() as io:
    if io.copyToTmp(args.weight):
        model_nn.load_weights(io.getTmpFileName(), by_name=True)
        print("...Uso i pesi già allenati...", flush=True)
    else:
        print("...NUOVO ALLENAMENTO...", flush=True)
print("", flush=True)

# esegue la generazione delle annotazioni con supporto del modello
genera_annotazioni(args.path_img, args.path_txt, model.isgrayscaled(), model.get_input_image_size())
