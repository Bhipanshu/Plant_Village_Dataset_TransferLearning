import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets



device = torch.device("cuda")

#Verifying CUDA
print(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = 'path/to/data_dir'

dataset = datasets.ImageFolder(data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Define data loaders for train and test sets
batch_size = 32  # You can adjust this based on your needs
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class CustomVGG16(nn.Module):
    def __init__(self, num_classes=15):
        super(CustomVGG16, self).__init__()
        
        # Load the pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)
        
        # Freeze all layers except the final classifier
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        # Modify the classifier
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, 15)

    def forward(self, x):
        return self.vgg16(x)


model = CustomVGG16()
model.to(device)

criterion = nn.CrossEntropyLoss()
criterion.to(device)
# optimizer = optim.SGD(model.vgg16.classifier.parameters(), lr=0.001, momentum=0.9)

optimizer = optim.SGD(model.vgg16.classifier.parameters(), lr=0.001, momentum=0.9)

num_epochs = 50
best_accuracy = 0.0
best_epoch = 0
best_model_state = None
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader))

    for i, data in progress_bar:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(inputs).to(device)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f} Accuracy: {accuracy * 100:.2f}%")
    
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch
        best_model_state = model.state_dict()

# Save the best model's state
best_model_filename = 'vgg16_individual_15classes.pth'
torch.save(best_model_state, best_model_filename)

print('Finished Training of VGG16')




model = CustomVGG16(num_classes=15)

model_path = 'vgg16_individual_15classes.pth'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.to(device)



#Flops and parameters
input_tensor = torch.randn(1, 3, 256, 256).to(device)
flops, _ = profile(model, inputs=(input_tensor,))
total_flops = flops / 10 ** 6  # Convert to MFLOPs
print(f"Total FLOPs: {total_flops:.2f} MFLOPs")




correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))



#Test for single image prediction

# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# image_path = '/media/bennettpc/fe11dcbb-f25e-4ef7-83dc-f1876b222064/Bhipanshu/VAE_vanilla/data_dir/Pepper__bell___Bacterial_spot/0bd0f439-013b-40ed-a6d1-4e67e971d437___JR_B.Spot 3272.JPG'
# image = Image.open(image_path).convert('RGB')

# input_tensor = preprocess(image)
# input_tensor = input_tensor.unsqueeze(0).to(device) # Add a batch dimension

# with torch.no_grad():
#     output = model(input_tensor)

# _, predicted_class = output.max(1)
# predicted_probability = torch.nn.functional.softmax(output, dim=1)[0] * 100

# class_names = {
#     0: 'Pepper__bell__Bacterial_spot',
#     1: 'Pepper__bell__healthy',
#     2: 'Potato__Early_blight',
#     3: 'Potato___healthy',
#     4: 'Potato__Late_blight',
#     5: 'Tomato__Target_Spot',
#     6: 'Tomato_Tomato_mosaic_virus',
#     7: 'Tomato_Tomato_YellowLeaf__Curl_Virus',
#     8: 'Tomato_Bacterial_Spot',
#     9: 'Tomato_Early_blight',
#     10: 'Tomato_healthy',
#     11: 'Tomato_Late_blight',
#     12: 'Tomato_Leaf_Mold',
#     13: 'Tomato_Septoria_leaf_spot',
#     14: 'Tomato_Spider_mites_Two_spotted_spider_mite'
# }


# predicted_class_name = class_names.get(predicted_class.item(), 'Unknown')

# print(f'Predicted Class: {predicted_class.item()}')
# print(f'Predicted Class Name: {predicted_class_name}')
# print(f'Predicted Probability: {predicted_probability[predicted_class.item()]:.2f}%')

