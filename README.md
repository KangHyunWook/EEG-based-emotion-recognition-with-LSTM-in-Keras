# EEG-based-emotion-recognition-with-LSTM-in-Keras

Program Description: https://www.youtube.com/watch?v=eVPcpiCcrL8&t=2s <br />

dataset link: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions?resource=download <br />
Place the downloaded file 'emotions.csv' to the working directory 

<h2>How to train</h2>

```
python train.py
```

Add --vis as option to see the visualization of the predicted emotional vectors on the train data

```
python train.py --vis --weights ./model/[saved_model_name]
```

You can add white noise data augmentation with --aug option, however performance degrades with eeg signal data unlike audio data.

<h2>How to test</h2>

```
python test.py -w [saved_model_name]
```
