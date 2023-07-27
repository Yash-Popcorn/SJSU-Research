import os
import json
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import h5py

# CHANGE YOUR PATH HERE
main_folder = "/home/mimi/Downloads/top10families"

# Create a dictionary to map family names to class indices
family_names = sorted(os.listdir(main_folder))
family_to_class = {family_name: idx for idx, family_name in enumerate(family_names)}

num_classes = len(family_names)

# Convert JSON to HDF5
def convert_json_to_hdf5(data_folder, hdf5_file_path):
    file_paths = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json'):
                file_paths.append(os.path.join(root, file))
                
    with h5py.File(hdf5_file_path, 'w') as f:
        seq_dataset = f.create_dataset('sequences', (len(file_paths), 1600), dtype='f')
        labels_dataset = f.create_dataset('labels', (len(file_paths), ), dtype='i')

        for i, path in enumerate(file_paths):
            with open(path, 'r') as json_file:
                json_data = json.load(json_file)
            sequence = [int(x, 16) / 255.0 for x in json_data[:1600]]
            seq_dataset[i] = sequence
            family_name = os.path.basename(os.path.dirname(path))
            family_label = family_to_class[family_name]
            labels_dataset[i] = family_label

# Create HDF5 file
# convert_json_to_hdf5(main_folder, 'data.hdf5')

# Split data into training and validation sets
with h5py.File('data.hdf5', 'r') as f:
    num_samples = len(f['sequences'])
train_indices, val_indices = train_test_split(np.arange(num_samples), test_size=0.2, random_state=42)

def data_generator(hdf5_file_path, indices, batch_size=32):
    with h5py.File(hdf5_file_path, 'r') as f:
        seq_dataset = f['sequences']
        labels_dataset = f['labels']
        
        while True:
            batch_indices = np.sort(np.random.choice(indices, size=batch_size, replace=False))
            batch_data = seq_dataset[batch_indices]
            batch_labels = labels_dataset[batch_indices]
            batch_labels_one_hot = tf.keras.utils.to_categorical(batch_labels, num_classes)
            yield batch_data, batch_labels_one_hot


# Sequence-to-Sequence Model
model = Sequential()
model.add(LSTM(512, input_shape=(1600, 1), recurrent_dropout=0.2))  # Adding dropout to LSTM layer
model.add(Dense(128, activation='tanh')) 
# model.add(Dense(64, activation='tanh'))  
model.add(Dense(num_classes, activation='softmax'))  # Output layer for classification

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Model Training
batch_size = 64
train_generator = data_generator("data.hdf5", train_indices, batch_size=batch_size)
val_generator = data_generator("data.hdf5", val_indices, batch_size=batch_size)

num_epochs = 10
train_steps_per_epoch = len(train_indices) // batch_size
val_steps_per_epoch = len(val_indices) // batch_size

model.fit(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=num_epochs,
                    validation_data=val_generator, validation_steps=val_steps_per_epoch)

# # Evaluate the model
loss, accuracy = model.evaluate(val_generator, steps=val_steps_per_epoch)
 
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
