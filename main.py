import kagglehub

#Download the data 
path = kagglehub.dataset_download("shyamgupta196/bone-fracture-split-classification")


# Classfiction for the data
class_to_idx = {
    "Avulsion fracture": 0,
    "Comminuted fracture": 1,
    "Compression-Crush fracture": 2,
    "Fracture Dislocation": 3,
    "Greenstick fracture": 4,
    "Hairline Fracture": 5,
    "Impacted fracture": 6,
    "Intra-articular fracture": 7,
    "Longitudinal fracture": 8,
    "Oblique fracture": 9,
    "Pathological fracture": 10,
    "Spiral Fracture": 11
}



from torch.utils.data import Dataset
from PIL import Image
import glob



#Buliding the Dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir,split='train' , transform=None):
        self.root_dir = root_dir  # Dataset path
        self.transform = transform  # Transformations
        self.class_labels = class_to_idx
        self.split = split

#{v: k for k, v in class_to_idx.items()}

        # Get all image paths
        self.image_paths = []
        self.labels = []
        for class_name, label in self.class_labels.items():

            class_images = glob.glob(root_dir + "/" + split + "/*/*.png" )  # Find all images

            self.image_paths.extend(class_images)
            self.labels.extend([label] * len(class_images))  # Assign labels

    def __len__(self):
        return len(self.image_paths)  # Total number of images

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]  # Get image path
        label = self.labels[idx]  # Get label

        # Load image using PIL
        image = Image.open(image_path)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, label  # Return processed image & label
    
    
    


dataset = CustomDataset(path)

print(len(dataset))

import matplotlib.pyplot as plt
import numpy as np

# Define mean & std for denormalization (EfficientNet Preprocessing



for i in range(5):
    img, label = dataset[i]  # Load image & label

    img

    print(type(img))



# Choosing the model
from torchvision.models import efficientnet_v2_m
import torch
import torch.nn as nn

model = efficientnet_v2_m(pretrained = True)

# Modify the classifier for binary classification (cats vs. dogs)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 11)  # 2 classes

print(model)



import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm



# 🔹 Training Loop
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, masks in tqdm(dataloader):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 🔹 Validation Loop
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

    return total_loss / len(dataloader)




# Runing the model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


image_transforms = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.PILToTensor(),

    transforms.Grayscale(), # the problem is here and  i dnk wy

    ])

# Load training dataset
train_dataset = CustomDataset(root_dir=path , transform=image_transforms)

# Load testing dataset
test_dataset = CustomDataset(root_dir=path, split="test" , transform=image_transforms)


# Create Train & Test DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)



import torch
from torch import nn
# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

num_epochs = 10  # Define number of epochs
train_losses = []
val_losses = []

# Training Loop
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, test_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
