# Face_Identification

The process of identification goes as follows:
1. Capture image from webcam, detect face in that image and crop the section of image containing the face
2. Use a pretrained model for classifying the image into binary classes: a) Chhitij b) Not Chhitij
3. The image and result is then shown on the screen

Epoch 100/100
125/125 [==============================] - 317s 3s/step - loss: 0.0322 - acc: 0.9980 - val_loss: 7.1321 - val_acc: 0.5526

This is what the last epoch data looks like. The validation accuracy is too low.
