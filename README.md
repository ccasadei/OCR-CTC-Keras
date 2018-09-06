**OCR con Keras/TF su MLEngine**

`by Cristiano Casadei`

**Glossario**

-   **GT**: Ground Truth, cioè il valore atteso
-   **PR**: Prediction, cioè il valore predetto dalla rete neurale 
-   **Val. Loss**: valore della funzione **loss** di validazione, sulla quale vengono salvati i migliori pesi di training
-   **Acc. Media**: accuracy media delle date complete, calcolata sul totale delle date.
    <br/>
    Aumenta quando una data viene correttamente predetta nella sua interezza.

    Esempio:
    <br/>
    data GT **01/23/4567**
    <br/>
    data PR **01/23/4567** &rarr; data correttamente predetta nella sua interezza
    
    data GT **01/23/4567**
    <br/>
    data PR **01/22/4567** &rarr; data non correttamente predetta nella sua interezza
    
-   **Acc. Media Cifre**: accuracy media delle cifre predette correttamente, calcolata sul totale delle cifre di tutte le date.
    <br/>
    Aumenta quando in una data una lettera viene correttamente predetta nella posizione giusta.
    
    Esempio:
    <br/>
    data GR **01/23/4567**
    <br/>
    data PR **01/22/5467** &rarr; 7 cifre predette correttamente nella loro posizione

**NOTA**: i neuroni di tipo GRU sembrano funzionare meglio di quelli LSTM, a parità di **Val. Loss**
    
____

**Risultati su Dataset Autogenerato MNIST con rumore SPECKLE**

**LSTM**

| Val. Loss | Acc. Media | Acc. Media Cifre |
| ---------:| ----------:| ----------------:|
| 4.507 | 0.2060 | 0.8322 |
| 3.809 | 0.2720 | 0.8603 |
| 3.029 | 0.3810 | 0.8970 |
| 2.769 | 0.4130 | 0.9016 |

**GRU**

| Val. Loss | Acc. Media | Acc. Media Cifre |
| ---------:| ----------:| ----------------:|
| 13.499 | 0.0000 | 0.4448 |
| 10.666 | 0.0070 | 0.5599 |
| 3.279 | 0.3950 | 0.8860 |
| 2.908 | 0.4420 | 0.8981 |

____

**Risultati su Dataset reale di "Service"**

**LSTM**

| Val. Loss | Acc. Media | Acc. Media Cifre |
| ---------:| ----------:| ----------------:|
| 6.336 | 0.1202 | 0.5652 |

**GRU**

| Val. Loss | Acc. Media | Acc. Media Cifre |
| ---------:| ----------:| ----------------:|
| 6.736 | 0.2568 | 0.6558 |
| 3.844 | 0.3934 | 0.7478 |
| 3.095 | 0.4900 | 0.7938 |
| 2.456 | 0.5400 | 0.8643 |
| 2.433 | 0.5860 | 0.8377 |
| 2.271 | 0.4860 | 0.7997 **dopo generalizzazione** |
| 2.221 | 0.6120 | 0.8457 |
| 2.212 | 0.6440 | 0.8569 |
| 1.678 | 0.4140 | 0.7326 **dopo generalizzazione**|

**GRU5 (versione simile a GRU ma con maggiore larghezza temporale e parte RNN semplificata)**
| Val. Loss | Acc. Media | Acc. Media Cifre |
| ---------:| ----------:| ----------------:|
| 3.747 | 0.2160 | 0.5810 |


**NOTA**: il modello GRU è stato selezionato come il più promettente

____

**Risultati su Dataset reale di "Service" + Autogenerato UNIPEN con rumore S&P**

**GRU**

| Val. Loss | Acc. Media | Acc. Media Cifre |
| ---------:| ----------:| ----------------:|
| 4.028 | 0.4414 | 0.7750 |
| 3.724 | 0.4271 | 0.7803 |
| 3.447 | 0.4600 | 0.8012 |
| 1.234 | 0.3771 | 0.7709 |

____

**NOTA**: ho provato ad usare sulla rete GRU dei features extractor diversi (a colori rgb), come VGG19, RESNET50 e DENSENET201, ma i risultati non sono migliorati significativamente, anzi in alcuni casi sono leggermente peggiorati (a parità di **Val. Loss**)

**Risultati su Dataset reale di "Service"**

**GRU2 (VGG19)**

| Val. Loss | Acc. Media | Acc. Media Cifre |
| ---------:| ----------:| ----------------:|
| 6.160 | 0.1311 | 0.5746 |

**GRU3 (RESNET50)**

| Val. Loss | Acc. Media | Acc. Media Cifre |
| ---------:| ----------:| ----------------:|
| 6.846 | 0.0984 | 0.5739 |

**GRU4 (DENSENET201)**

| Val. Loss | Acc. Media | Acc. Media Cifre |
| ---------:| ----------:| ----------------:|
| 7.235 | 0.1913 | 0.5797 |
| 6.250 | 0.1913 | 0.6391 |
| 5.537 | 0.2022 | 0.6254 |

