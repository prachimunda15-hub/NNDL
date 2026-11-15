Speech Emotion Recognition on CREMA-D

The goal of this project is to classify the emotion expressed in short speech clips. The classification is performed using a CNN + BiLSTM + attention network. The network is then tested on recordings of actors which were not used for training.

This project was done as part of the course EC3608 Neural Networks and Deep Learning.

Results
Model	Test accuracy
CNN + BiLSTM + attention (final model, used in the web app)	57.93%
Random guessing would result in 16.7% accuracy for the 6 different emotions. The model performs 3.5 times better than random guessing.
The test set is actor-independent, i.e. no actor is used in train, validation and test.
These are results for a single run on acted studio speech. Baseline scores for CNN + BiLSTM and CNN + attention on CREMA-D are still to be added.
Dataset

CREMA-D: acted speech from 91 actors reading 12 sentences with different emotions.

There are 6 classes: Angry, Disgust, Fear, Happy, Neutral, Sad. Files have names like 1001_DFA_ANG_XX.wav. The actor ID and emotion can be retrieved from the file name. Since there are fewer clips for Neutral than for the other classes, training is performed with balanced class weights.

Citation: Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. IEEE Transactions on Affective Computing.

How it works
1. Preprocessing (web app and final model)
Step	Setting
Audio	16 kHz, mono, cropped or zero-padded to 5 seconds
STFT	400-sample frames (25 ms), 160-sample hop (10 ms), 512-point FFT
Spectrogram	Power spectrogram, then 128 mel bands from 80 Hz to 7,600 Hz
Scaling	Natural log (log(mel + 1e-6)), then standardised per recording
Output shape	500 x 128 x 1

The preprocessing code is used in preprocessing.py and in app.py. It is important that the same preprocessing is used during training and inference. Otherwise the results are not reliable.

cnn_bilstm.py uses a different librosa pipeline (16 kHz, 128 mel bands, n_fft=1024, hop 256, decibels scaled to 0-1, 200 frames). The resulting input is 200 x 128 x 1.

2. Models
CNN + BiLSTM (cnn_bilstm.py): 4 conv blocks (32, 64, 128, 128 filters) with BatchNorm and max pooling, 2 BiLSTM layers (128 and 64 units), dense layer and softmax.
CNN + attention (cnn_attention.py): 3 conv blocks (32, 64, 128 filters) with BatchNorm, channel attention, global average pooling, 2 dense layers.
CNN + BiLSTM + attention (final model): CNN for feature extraction, custom FrequencyPooling layer (mean over frequency axis), BiLSTM, multi-head attention, feed forward layer, average and max pooling, dense softmax layer.

In all models, the CNN is used for feature extraction. The rest of the model defines what is important.

3. Training setup (cnn_bilstm.py)
Speaker-independent split with GroupShuffleSplit: about 70% train, 15% validation, 15% test. It is checked that no actor is used in more than one of the three sets.
Adam, lr=5e-4, batch_size=16, up to 80 epochs, seed=42
Balanced class weights
Early stopping with validation loss (patience 12, restores best weights), halve lr after 5 flat epochs, best checkpoint saved by validation accuracy
Outputs: classification report, macro and weighted F1, confusion matrix, training curves
Web app

The Streamlit app (app.py) allows you to upload an audio file (wav, mp3, flac, ogg, m4a). It will preprocess the clip, use the model to predict the emotion, output confidence, probability chart, top 3 predictions and results table.

cnn_bilstm_attention.py is another, simpler Streamlit page which allows you to upload a preprocessed .npy feature array with shape (500, 128, 1) instead of the raw audio.

Getting started
bash

cd NNDL
pip install tensorflow librosa streamlit pandas numpy matplotlib scikit-learn

Place the trained model at the expected location for app.py (next to app.py):

CREMA-CNN-BILSTM-ATTENTION-RESULTS/best_crema_bilstm_attention.keras

Then run:

bash
streamlit run app.py
git clone https://github.com/prachimunda15-hub/NNDL
For training the CNN + BiLSTM baseline, please set DATASET_PATH at the top of cnn_bilstm.py to your CREMA-D .wav folder. Then run:

bash
python cnn_bilstm.py

Features are cached as .npy files, so the second run will be much faster.

Repository structure
app.py                      Streamlit app for raw audio
preprocessing.py            Audio to log-mel features (500 x 128 x 1)
cnn_bilstm.py               CNN + BiLSTM training and evaluation on CREMA-D
cnn_attention.py            CNN + channel attention model
cnn_bilstm_attention.py     Streamlit page for .npy features
Limitations
Only acted studio speech. No results on real world or clinical use yet.
Single run on a single actor split. It would be better to repeat the run on several seeds and actor folds to get a more reliable number.
The probabilities are the output of the model, not a calibrated certainty. Noise, recording quality, speaking style can affect the result.
cnn_attention.py still has NUM_CLASSES = 8 and 551 frames input (from RAVDESS). Please update both for CREMA-D before running the code.
Possible next steps
Trim silence and mask padded frames.
Augmentation (noise, pitch shift, SpecAugment)
Results per emotion, check confusion matrix
Test on RAVDESS, noisy real world audio
