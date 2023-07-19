import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import seaborn as sns
import matplotlib.pyplot as plt
import random
from torchvision.transforms import ToPILImage
from torch import nn, optim
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load your csv
df = pd.read_csv(r'F:\DATASET\model(143x143).csv')
img = r'F:\DATASET\model(143x143)'
save_folder = 'F:\DATASET'
img_folder = r'F:\DATASET\model(143x143)'

df_sorted = df.sort_values(by='image_id', ascending=True)  
print(df_sorted.label)

train_ratio = 0.7  
val_ratio = 0.15  
test_ratio = 0.15 

train_df, test_df = train_test_split(df_sorted, test_size=test_ratio, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)

# Reset the index of the train, validation, and test DataFrames if needed
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Define transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'augmentation': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Function to load images
def load_image(img_path, transform=None):
    image = Image.open(img_path).convert('RGB')
    if transform:
        image = transform(image)
    return image
# Load images
train_images = []
for img in train_df.iloc[:, 0]:
    image_path = os.path.join(img_folder, img + '.png')
    train_image = load_image(image_path, transform=data_transforms['train'])
    train_images.append(train_image)
    print('Count:', len(train_images))
    
print("Number of train images:", len(train_images))

val_images = []
for img in val_df.iloc[:, 0]:
    image_path = os.path.join(img_folder, img + '.png')
    val_image = load_image(image_path, transform=data_transforms['val'])
    val_images.append(val_image)
    print('Count:', len(val_images))
    
print("Number of validation images:", len(val_images))

test_images = []
for img in test_df.iloc[:, 0]:
    image_path = os.path.join(img_folder, img + '.png')
    test_image = load_image(image_path, transform=data_transforms['val'])
    test_images.append(test_image)
    print('Count:', len(test_images))
    
print("Number of test images:", len(test_images))

# Define labels
label_mapping = {'CE': 0, 'LAA': 1}
train_labels = [label_mapping[label.upper()] for label in train_df.iloc[:, -1].tolist()]
val_labels = [label_mapping[label.upper()] for label in val_df.iloc[:, -1].tolist()]
test_labels = [label_mapping[label.upper()] for label in test_df.iloc[:, -1].tolist()]

num_classes = len(label_mapping)

# Count the number of CE and LAA labels in the train, validation, and test sets
train_ce_count = train_labels.count(0)
train_laa_count = train_labels.count(1)
val_ce_count = val_labels.count(0)
val_laa_count = val_labels.count(1)
test_ce_count = test_labels.count(0)
test_laa_count = test_labels.count(1)

print("Train - CE:", train_ce_count)
print("Train - LAA:", train_laa_count)
print("Validation - CE:", val_ce_count)
print("Validation - LAA:", val_laa_count)
print("Test - CE:", test_ce_count)
print("Test - LAA:", test_laa_count)

# Count the number of labels in the train, validation, and test sets
train_counts = [train_labels.count(i) for i in range(num_classes)]
val_counts = [val_labels.count(i) for i in range(num_classes)]
test_counts = [test_labels.count(i) for i in range(num_classes)]

desired_counts = {'LAA': 400}
to_pil = ToPILImage()
laa_augmented = []

for i in range(len(train_labels)):
    print(train_labels[i])
    label = train_labels[i]
    image = train_images[i]
    pil_image = to_pil(image)

    
    if label == 1 and train_counts[label] < desired_counts['LAA']:
        for _ in range(desired_counts['LAA'] - train_counts[label]):
            random_index = random.randint(0, len(train_images) - 1)
            random_image = train_images[random_index]
            augmented_image = data_transforms['augmentation'](to_pil(random_image))
            laa_augmented.append(augmented_image)
        break

train_images_augmented = train_images + laa_augmented
train_labels_augmented = train_labels + [1] * len(laa_augmented)

combined_data = list(zip(train_images_augmented, train_labels_augmented))
random.shuffle(combined_data)
train_images_augmented, train_labels_augmented = zip(*combined_data)

train_images_augmented = [image for image in train_images_augmented]
train_labels_augmented = list(train_labels_augmented)


# Load pre-trained model
model = models.resnet50(pretrained=True)

# Modify the last layer of the ResNet-50 model
num_ftrs = model.fc.in_features
num_classes = len(set(train_labels))  # number of unique labels in the training data
model.fc = nn.Linear(num_ftrs, num_classes)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the batch size
batch_size = 32

# Define the datasets
train_data = [(img, label) for img, label in zip(train_images_augmented, train_labels_augmented)]
val_data = [(img, label) for img, label in zip(val_images, val_labels)]
test_data = [(img, label) for img, label in zip(test_images, test_labels)]

# Define the data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Train the model
model.train()
for epoch in range(20):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")

# Validate the model
model.eval()
with torch.no_grad():
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct / total

print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Test the model
model.eval()
with torch.no_grad():
    running_loss = 0.0
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predicted_labels.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct / total

# Calculate precision, recall, and F1 score collectively
overall_precision = precision_score(true_labels, predicted_labels, average='weighted')
overall_recall = recall_score(true_labels, predicted_labels, average='weighted')
overall_f1 = f1_score(true_labels, predicted_labels, average='weighted')

# Calculate precision, recall, and F1 score for each class
class_precision = precision_score(true_labels, predicted_labels, average=None)
class_recall = recall_score(true_labels, predicted_labels, average=None)
class_f1 = f1_score(true_labels, predicted_labels, average=None)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("Overall Precision:", overall_precision)
print("Overall Recall:", overall_recall)
print("Overall F1 Score:", overall_f1)
print("Class 0 (CE) - Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(class_precision[0], class_recall[0], class_f1[0]))
print("Class 1 (LAA) - Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(class_precision[1], class_recall[1], class_f1[1]))

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig(os.path.join(save_folder, "confusion_matrix.png"))  # Save the confusion matrix as an image file
plt.show()

print("Confusion Matrix:\n", cm)
torch.save(model.state_dict(), 'modelcnn.pth')
