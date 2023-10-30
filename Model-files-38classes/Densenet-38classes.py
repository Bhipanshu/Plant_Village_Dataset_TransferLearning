import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torchvision.models as models
import torch.nn as nn
from thop import profile 
import time
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets



device = torch.device("cuda") 
print(device)

#Data Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


data_dir = 'Data folder path'

dataset = datasets.ImageFolder(data_dir,transform=transform)
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset,test_dataset = random_split(dataset,[train_size,test_size])
batch_size = 64
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)



#Customized pretrained model
class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=38):
        super(CustomDenseNet, self).__init__()

        self.densenet = models.densenet121(pretrained=True)

        for param in self.densenet.parameters():
            param.requires_grad = False

        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.densenet(x)
        return x
    
    
    

model = CustomDenseNet()
model.to(device)



#loss and optimization 
criterion = nn.CrossEntropyLoss()
criterion.to(device)

optimizer = optim.SGD(model.densenet.classifier.parameters(), lr=0.001, momentum=0.9)

num_epochs = 50
best_accuracy = 0.0
best_epoch = 0
best_model_state = None
total_training_time = 0.0

for epoch in range(num_epochs):
    epoch_start_time = time.time()

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

    epoch_end_time = time.time()
    epoch_elapsed_time = epoch_end_time - epoch_start_time
    total_training_time += epoch_elapsed_time
total_training_time_minutes = total_training_time / 60



best_model_filename = 'densenet-38classes.pth'
torch.save(best_model_state, best_model_filename)
print(f"Total Training Time: {total_training_time_minutes:.2f} minutes")

print('Finished Training of Densenet')




model = CustomDenseNet()
model_path = 'densenet-38classes.pth'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.to(device)


input_tensor = torch.randn(1, 3, 256, 256).to(device)
# Profile the model to calculate FLOPs
flops, params = profile(model, inputs=(input_tensor,))
total_flops = flops / 10 ** 6  
total_params = params / 10 ** 6
print(f"Total FLOPs: {total_flops:.2f} MFLOPs")
print(f'Total parameters: {total_params:.2f} Params(M)')


correct = 0
total = 0
total_testing_time = 0
start_time = time.time()
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
total_testing_time_minutes = elapsed_time / 60

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
print(f'Total testing time : {total_testing_time_minutes:.2f} mins')





#testing on single images
from PIL import Image

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_path = ''  # Replace with the path to your image
image = Image.open(image_path).convert('RGB')

input_tensor = preprocess(image)
input_tensor = input_tensor.unsqueeze(0).to(device) # Add a batch dimension

with torch.no_grad():
    output = model(input_tensor)

_, predicted_class = output.max(1)
predicted_probability = torch.nn.functional.softmax(output, dim=1)[0] * 100

index_to_class = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    4: "Blueberry___healthy",
    5: "Cherry_(including_sour)___Powdery_mildew",
    6: "Cherry_(including_sour)___healthy",
    7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    8: "Corn_(maize)___Common_rust_",
    9: "Corn_(maize)___Northern_Leaf_Blight",
    10: "Corn_(maize)___healthy",
    11: "Grape___Black_rot",
    12: "Grape___Esca_(Black_Measles)",
    13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    14: "Grape___healthy",
    15: "Orange___Haunglongbing_(Citrus_greening)",
    16: "Peach___Bacterial_spot",
    17: "Peach___healthy",
    18: "Pepper,_bell___Bacterial_spot",
    19: "Pepper,_bell___healthy",
    20: "Potato___Early_blight",
    21: "Potato___Late_blight",
    22: "Potato___healthy",
    23: "Raspberry___healthy",
    24: "Soybean___healthy",
    25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch",
    27: "Strawberry___healthy",
    28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight",
    30: "Tomato___Late_blight",
    31: "Tomato___Leaf_Mold",
    32: "Tomato___Septoria_leaf_spot",
    33: "Tomato___Spider_mites Two-spotted_spider_mite",
    34: "Tomato___Target_Spot",
    35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    36: "Tomato___Tomato_mosaic_virus",
    37: "Tomato___healthy"
}


predicted_class_name = index_to_class.get(predicted_class.item(), 'Unknown')

print(f'Predicted Class: {predicted_class.item()}')
print(f'Predicted Class Name: {predicted_class_name}')
print(f'Predicted Probability: {predicted_probability[predicted_class.item()]:.2f}%')




    
    

    


