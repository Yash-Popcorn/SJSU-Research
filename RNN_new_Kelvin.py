import os
import json
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Step 2: Data Preparation
def collect_json_file_paths(data_folder):
    file_paths = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json'):
                file_paths.append(os.path.join(root, file))
    return file_paths
# CHANGE YOUR PATH HERE
main_folder = "/home/mimi/Downloads/top10families"
file_paths = collect_json_file_paths(main_folder)

# Split data into training and validation sets
train_file_paths, val_file_paths = train_test_split(file_paths, test_size=0.2, random_state=42)

# Create a dictionary to map family names to class indices
family_names = sorted(os.listdir(main_folder))
family_to_class = {family_name: idx for idx, family_name in enumerate(family_names)}

# Split data into training and validation sets
train_file_paths, val_file_paths = train_test_split(file_paths, test_size=0.2, random_state=42)

# Step 3: Data Generator

num_classes = len(family_names)

def data_generator(file_paths, batch_size=64):
    while True:
        batch_paths = np.random.choice(file_paths, size=batch_size, replace=False)
        batch_data = []
        batch_labels = []
        
        for path in batch_paths:
            with open(path, 'r') as f:
                json_data = json.load(f)
            sequence = [int(x, 16) / 255.0 for x in json_data[:2048]]  # Scale hex values to [0, 1]
            batch_data.append(sequence)
            # Extract the family name from the folder name
            family_name = os.path.basename(os.path.dirname(path))
            family_label = family_to_class[family_name]
            batch_labels.append(family_label)
            
        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)
        batch_labels_one_hot = tf.keras.utils.to_categorical(batch_labels, num_classes)
        yield batch_data, batch_labels_one_hot

# Step 4: Sequence-to-Sequence Model

model = Sequential()
model.add(LSTM(128, input_shape=(2048, 1), dropout=0.2, recurrent_dropout=0.2))  # Adding dropout to LSTM layer
model.add(Dense(256, activation='tanh'))  
model.add(Dense(num_classes, activation='softmax'))  # Output layer for classification

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Step 5: Model Training

batch_size = 64
train_generator = data_generator(train_file_paths, batch_size=batch_size)
val_generator = data_generator(val_file_paths, batch_size=batch_size)

num_epochs = 10
train_steps_per_epoch = len(train_file_paths) // batch_size
val_steps_per_epoch = len(val_file_paths) // batch_size

model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=num_epochs,
                    validation_data=val_generator, validation_steps=val_steps_per_epoch)