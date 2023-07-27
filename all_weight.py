import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import h5py
import struct

#CHANGE PATH HERE
main_folder = "/home/mimi/Downloads/top10families"

family_names = sorted(os.listdir(main_folder))
family_to_class = {family_name: idx for idx, family_name in enumerate(family_names)}

num_classes = len(family_names)

def modify_weights(weights, binary_data, num_bits_per_weight):
    binary_data_gen = (bit for bit in binary_data)
    bit_counter = 0

    for i in range(len(weights)):
        for j in range(len(weights[i])):
            flat_weight = weights[i][j].flatten()

            for k in range(len(flat_weight)):
                try:
                    bits = [next(binary_data_gen) for _ in range(num_bits_per_weight)]
                except StopIteration:
                    break
                
                binary_weight = struct.unpack('!I', struct.pack('!f', flat_weight[k]))[0]

                for bit_idx, bit in enumerate(bits):
                    binary_weight = (binary_weight & ~(1 << bit_idx)) | (int(bit) << bit_idx)
                    bit_counter += 1

                flat_weight[k] = struct.unpack('!f', struct.pack('!I', binary_weight))[0]

            weights[i][j] = flat_weight.reshape(weights[i][j].shape)

    return weights


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


file_path = '/home/mimi/Downloads/alice.txt'
with open(file_path, 'rb') as file:
    binary_data = list(bin(int.from_bytes(file.read(), 'big'))[2:])
weights = model.get_weights()
new_weights = modify_weights(weights, binary_data, 15)
model.set_weights(new_weights)

loss, accuracy = model.evaluate(val_generator, steps=val_steps_per_epoch)
 
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


