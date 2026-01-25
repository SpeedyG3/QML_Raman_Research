import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.models import load_model
import pennylane as qml
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
from braket.tracking import Tracker
from braket.aws import AwsDevice

# Load the encoder model and dataset
encoder_model = load_model("latest_latest.h5")
data = pd.read_csv('aug_weathered_5plastics_baselineCorr.csv')

# Preprocessing the data
scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize between 0 and 1
remove_classes = ["PS", "PC", "NC", "/", "ABS"]
data = data[~data['plastic'].isin(remove_classes)]
wave_numbers = data.columns[2:]
spectra = data.iloc[:, 2:]

all_data = []
for col in range(len(spectra)):
    y = spectra.iloc[col].values.astype(float)
    temp = y.reshape(-1, 1)

    temp = scaler.fit_transform(temp)
    temp = temp.reshape(1, -1)
    all_data.append(temp)

encoder = LabelEncoder()
labels = data['plastic'].astype(str)
X = np.array(all_data)
y = encoder.fit_transform(labels)

# Apply AutoEncoder to the data
AE_op = encoder_model.predict(X)
X_pca = X.squeeze(axis=1)
pca = PCA(n_components=16)
X_pca = pca.fit_transform(X_pca)
X_AE = np.squeeze(AE_op, axis=1)

# LDA transformation
lda = LinearDiscriminantAnalysis(n_components=3)
lda_ae = lda.fit_transform(X_AE, y)
lda2 = LinearDiscriminantAnalysis(n_components=3)
lda_pca = lda2.fit_transform(X_pca, y)

# Concatenate LDA features
vqc_data = np.concatenate((lda_ae, lda_pca), axis=1)

# Normalize data
scaler = MinMaxScaler()
vqc_data = scaler.fit_transform(vqc_data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(vqc_data, y, test_size=0.2, random_state=42, stratify=y)

# Convert data to torch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

# Start tracking AWS Braket task costs
t = Tracker().start()
device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
device = AwsDevice(device_arn)
dev = qml.device('braket.aws.qubit', device_arn=device_arn, wires=6, shots=100)

# Quantum Circuit with Optimized Speed
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(6), rotation="Y")
    for w in weights:
        for i in range(6):
            qml.RY(w[i], wires=i)
            qml.RZ(w[i], wires=i)
        for i in range(1, 6):
            qml.CNOT(wires=[0, i])
    return [qml.expval(qml.PauliZ(i)) for i in range(6)]

# Define the Quantum Classifier Model
class QuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(0.01 * torch.randn(1, 6))  # Trainable weights
        self.fc = nn.Linear(6, 5)  # Fully connected layer for 5 classes

    def forward(self, x):
        q_out = torch.stack([torch.tensor(quantum_circuit(xi, self.q_weights), dtype=torch.float32) for xi in x])
        return self.fc(q_out)  # Output from quantum circuit to final classification

# Hyperparameters
epochs = 150
batch_size = 16
learning_rate = 0.01

# Initialize Model, Loss, Optimizer
model = QuantumClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
loss_values = []
accuracy_values = []

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    correct = (y_pred.argmax(dim=1) == y_train).sum().item()
    accuracy = correct / len(y_train) * 100

    loss_values.append(loss.item())
    accuracy_values.append(accuracy)

    print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")
    print("Quantum Task Summary:", t.quantum_tasks_statistics())
    print(f"Estimated cost so far: {t.qpu_tasks_cost() + t.simulator_tasks_cost():.3f} USD\n")

# Evaluate on test set
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    predicted_labels_test = y_pred_test.argmax(dim=1)
    test_accuracy = (predicted_labels_test == y_test).float().mean().item() * 100

# Save the training metrics to CSV
train_metrics_df = pd.DataFrame({
    'Epoch': range(epochs),
    'Loss': loss_values,
    'Accuracy': accuracy_values
})
train_metrics_df.to_csv("train_metrics.csv", index=False)

# Save the classification report to a text file
report = classification_report(y_test.numpy(), predicted_labels_test.numpy(), target_names=encoder.classes_, digits=4)
with open("classification_report.txt", "w") as f:
    f.write("📌 Classification Report on Test Dataset:\n")
    f.write("-" * 70 + "\n")
    f.write(report)
    f.write("-" * 70 + "\n")

# Save the ROC curve plot
y_test_binarized = label_binarize(y_test.numpy(), classes=np.arange(5))
y_pred_prob = torch.softmax(y_pred_test, dim=1).numpy()

roc_plot_filename = "roc_curve.png"
plt.figure(figsize=(10, 8), dpi=300)
for i, color in zip(range(5), ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{encoder.classes_[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier (AUC = 0.50)")
plt.xlabel("False Positive Rate", fontsize=14, fontweight="bold", labelpad=10)
plt.ylabel("True Positive Rate", fontsize=14, fontweight="bold", labelpad=10)
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight="bold", pad=15)
plt.legend(fontsize=12, loc="lower right", frameon=True, shadow=True, fancybox=True)
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(roc_plot_filename)

# Save the confusion matrix plot
conf_matrix = confusion_matrix(y_test.numpy(), predicted_labels_test.numpy())
conf_matrix_filename = "confusion_matrix.png"
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="viridis", xticklabels=encoder.classes_,
            yticklabels=encoder.classes_, linewidths=0.5, linecolor="gray", cbar=True, annot_kws={"size": 14})
plt.xlabel("Predicted Label", fontsize=14, fontweight="bold", labelpad=10)
plt.ylabel("True Label", fontsize=14, fontweight="bold", labelpad=10)
plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=0)
plt.savefig(conf_matrix_filename)

# Save the quantum circuit diagram
quantum_circuit_filename = "quantum_circuit_diagram.png"
fig, ax = qml.draw_mpl(quantum_circuit)(torch.rand(6), model.q_weights.detach().numpy())
fig.savefig(quantum_circuit_filename)

# Save model weights
weights_filename = "model_weights.npy"
np.save(weights_filename, model.q_weights.detach().numpy())

# Print final accuracy and save outputs
print("=" * 70)
print(f"🚀 Final Test Accuracy: {test_accuracy:.2f}%")
print("=" * 70)
print(f"Saving training metrics to 'train_metrics.csv'.")
print(f"Saving classification report to 'classification_report.txt'.")
print(f"Saving ROC Curve to 'roc_curve.png'.")
print(f"Saving Confusion Matrix to 'confusion_matrix.png'.")
print(f"Saving Quantum Circuit Diagram to 'quantum_circuit_diagram.png'.")
print(f"Saving Model Weights to 'model_weights.npy'.")
