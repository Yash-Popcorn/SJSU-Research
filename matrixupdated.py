import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

true_classes = []
predicted_classes = []

for step in range(val_steps_per_epoch):
    batch_data, batch_labels = next(val_generator)
    batch_predictions = model.predict(batch_data)
    true_classes.extend(np.argmax(batch_labels, axis=1))
    predicted_classes.extend(np.argmax(batch_predictions, axis=1))

predicted_classes = np.array(predicted_classes)

# Create the confusion matrix
confusion_mtx = confusion_matrix(true_classes, predicted_classes)
cm_normalized = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]

class_labels = ['adload', 'BHO', 'ceeinject', 'onlinegames', 'renos', 'startpage', 'vb', 'vbinject', 'vobfus', 'winwebsec']

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(cm_normalized, cmap='Blues')
cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(class_labels)))
ax.set_yticks(np.arange(len(class_labels)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        text = ax.text(j, i, f"{cm_normalized[i, j]:.2%}",
                       ha="center", va="center", color="black")

ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

plt.show()

# Print classification report
target_names = [f'Class {i}' for i in range(num_classes)]
print(classification_report(true_classes, predicted_classes, target_names=target_names))
