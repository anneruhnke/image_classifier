# Import libraries
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import argparse

# Build argument parser
ap = argparse.ArgumentParser()

ap.add_argument('-dir', required=True, help="image directory", type = str)
ap.add_argument('-out', required=True, help="# output units", type = int)

ap.add_argument('-dict', required=False, default='cat_to_name.json')
ap.add_argument('-arch', required=False, default="vgg11", help="pretrained model architecture; default = 'vgg11', or 'densenet121'", type = str)
ap.add_argument('-ep', required=False, default=10, help="# training iterations; default=10", type = int)
ap.add_argument('-lr', required=False, default=0.003, help="learning rate; default=0.003", type = float)
ap.add_argument('-hidden', required=False, default=4096, help="# hidden units; default=4096", type = int)
ap.add_argument('-dev', required=False, default ='gpu', help="'GPU' or 'CPU'", type = str)

args = vars(ap.parse_args())
print(args)

# Load data
print("\nImage directory: {}".format(args['dir']))
print("The following subfolders are required: /train, /valid, /test")

data_dir = args['dir']
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

# Map labels
with open(args['dict'], 'r') as f:
    cat_to_name = json.load(f)

# Build and train your network

print("\nThe model is ", args['arch']) 

if args['arch'] =='vgg11':
	model = models.vgg11(pretrained=True)
else:
	model = models.densenet121(pretrained=True)

# GPU or CPU
if args['dev'] == 'gpu' and torch.cuda.is_available():
    device = "cuda"
    print("\nGPU available")
elif args['dev'] == 'gpu' and torch.cuda.is_available() == False:
    print("\nGPU not available; switching to CPU")
    device = "cpu"
elif args['dev'] == 'cpu':
    device = "cpu"
    
# Freeze parameters 
for param in model.parameters():
    param.requires_grad = False

# Define new untrained feed-forward network as classifier    
if args['arch'] == 'vgg11':
    l_in = 25088 
else:
    l_in = 1024

l_hidden = int(args['hidden'])
l_out = int(args['out'])

model.classifier = nn.Sequential(nn.Linear(l_in, l_hidden),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3,),
                                 nn.Linear(l_hidden,l_out),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=float(args['lr']))
model.to(device)

# Train model
print("\nTrain the model")

epochs = int(args['ep'])
steps = 0
running_loss = 0
print_every = 500

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

# Test model
print("\nTest the model")

model.eval()
test_loss = 0
accuracy = 0
        
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        test_loss += criterion(logps, labels)
                
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy +=torch.mean(equals.type(torch.FloatTensor))
            
    print ("Test loss: {:.3f}..".format(test_loss/len(testloader)),
           "Test accuracy:{:.3f}..".format(accuracy/len(testloader)))
    
# Save the checkpoint
model.class_to_idx = train_data.class_to_idx

checkpoint={'architecture': args['arch'],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'classifier': model.classifier,
            'class_label': train_data.class_to_idx,
            'epochs': int(args['ep']),
            'learning_rate': args['lr']}

torch.save(checkpoint, 'checkpoint.pth')

print("\nModel is saved.")