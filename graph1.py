import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="cnn_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input details to understand the expected input shape
print("Input details:", input_details)

def load_images_from_folders(base_folder):
    images = []
    labels = []
    label_dict = {"drowsy": 0, "active": 1}
    
    for label_name, label in label_dict.items():
        folder = os.path.join(base_folder, label_name)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))  # Resize to model's input size
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (if needed)
                img = img.astype(np.float32) / 255.0  # Normalize pixel values
                images.append(img)
                labels.append(label)
                
    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_images_from_folders("C:\\Users\\gchit_7t0yidw\\OneDrive\\Desktop\\mlnew\\predic")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def predict(interpreter, input_details, output_details, images):
    input_shape = input_details[0]['shape']
    predictions = []
    for img in images:
        # Ensure input shape matches model's expected shape
        input_data = np.expand_dims(img, axis=0)
        input_data = input_data.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output_data))
    return np.array(predictions)

# Make predictions
y_pred = predict(interpreter, input_details, output_details, X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Drowsy", "Active"])
disp.plot()
plt.title('Confusion Matrix')
plt.show()

# Precision-Recall Curve
y_test_bin = np.eye(len(np.unique(y_test)))[y_test]  # One-hot encoding
y_pred_bin = np.eye(len(np.unique(y_pred)))[y_pred]

precision = dict()
recall = dict()
for i in range(len(np.unique(y_test))):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_bin[:, i])

# Plot Precision-Recall for each class
for i in range(len(np.unique(y_test))):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {i}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()

# ROC Curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(np.unique(y_test))):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC for each class
for i in range(len(np.unique(y_test))):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
