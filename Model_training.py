import tensorflow as tf
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot
from sklearn.model_selection import train_test_split


df = pd.read_csv(r'path/to/the/dataset')

#dictionary of labels and which expression they depict
labels_of_test = {0:"Anger", 1:"Disgust", 2:"Fear", 3:"Happiness", 4:"Sadness", 5:"Surprise", 6:"Neutral"}

#pixels of single image as (48,48) numpy array
# (np.array(df.pixels.loc[0].split(' ')).reshape(48,48).astype('float'))

# pass the complete dataset
img_array = df.pixels.apply(lambda x : np.array(x.split(' ')).reshape(48,48,1).astype('float32'))

#shape of the array
img_array = np.stack(img_array, axis=0)
#print(img_array.shape)

labels = df.emotion.values
# print(labels)

#splitting the test and training data
X_train,X_test,y_train,y_test = train_test_split(img_array, labels, test_size = 0.15)
#print(X_train.shape)

#normalisation
X_train = X_train/255
X_test = X_test/255

# model defination
recog_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation='relu', input_shape = (48,48,1)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),


        tf.keras.layers.Conv2D(64,(3,3),activation='relu', input_shape = (48,48,1)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),



        tf.keras.layers.Conv2D(128,(3,3),activation='relu', input_shape = (48,48,1)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),



        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(7, activation = 'softmax')
])


# model summary
#print(recog_model.summary())

# model compilation
recog_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003),
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])

# saving the model
try:
    os.mkdir(r'path/of/location/you/want/your/directory')
except:
    pass

file_name = 'best_model.h5'
checkpoint_path = os.path.join(r'path/of/location/you/want/your/directory', file_name)

call_back = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                               monitor = 'val_accuracy',
                                               verbose =1,
                                               save_freq = 'epoch',
                                               save_best_only = True,
                                               save_weights_only = False,
                                               mode = 'max')

# model training
recog_model.fit(X_train,y_train,epochs = 20, validation_split = 0.2, callbacks = call_back)


#Showing the predictions

final_model = tf.keras.models.load_model(r'path/to/your/model')
from IPython.display import clear_output
correct = 0
check = 0
for i in range(3000):
    print(f'actual label is {labels_of_test[y_test[i]]}')
    predicted_class = final_model.predict(tf.expand_dims(X_test[i],0)).argmax()
    print(f'predicted label is {labels_of_test[predicted_class]}')
    if  labels_of_test[y_test[i]] == labels_of_test[predicted_class] :
        correct = correct+1
    check = check+1

    pyplot.imshow(X_test[i].reshape((48,48)))

    pyplot.show(block = False)
    pyplot.pause(0.2)
    pyplot.close()


#accuracy of test set
accuracy = (correct/check) * 100
print("Accuracy of model on testing set is ", accuracy)
