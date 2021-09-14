# Import libraries
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models
import json
import argparse

# Build argument parser
ap = argparse.ArgumentParser()

ap.add_argument('-pic', required=True, help='Image path', type = str)

ap.add_argument('-dict', required=False, default='cat_to_name.json', type = str)
ap.add_argument('-dev', required=False, default='cpu', help="'GPU' or 'CPU'", type = str)
ap.add_argument('-K', required=False, default=5, help='# top classes', type = int)
                
args = vars(ap.parse_args())

image_path = args['pic']
if args['dev'] == 'gpu' and torch.cuda.is_available():
    device = "cuda"
elif args['dev'] == 'gpu' and torch.cuda.is_available() == False:
    device = "cpu"
elif args['dev'] == 'cpu':
    device = "cpu"

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    
    if device == 'cpu':
        fn = torch.load(path, map_location=str(device))
    else:
        fn = torch.load(path)
    
    classifier = fn['classifier']

#    print(fn['architecture'])
    if fn['architecture'] == "vgg11":
        model = models.vgg11(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
        
    model.classifier = classifier
    model.load_state_dict(fn['state_dict'])
    class_to_idx = fn['class_label']
     
    return model, class_to_idx

# Preprocessing image 
def process_image(image):
    im = Image.open(image)
    im.thumbnail((256, 256))
    
    width, height = im.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = left + 224
    bottom = top + 224
    im = im.crop((left, top, right, bottom))
    
    np_image = np.array(im) /255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    image = torch.from_numpy(np_image)
    
    return image

# Define predict function
def predict(image_path, model, topk):
    
    with torch.set_grad_enabled(False):
        model.to(device)
        
        if device == "cpu":
            image = process_image(image_path).unsqueeze_(0).type(torch.FloatTensor).to(device)
        else:
            image = process_image(image_path).unsqueeze_(0).type(torch.cuda.FloatTensor)

        logps = model.forward(image)            
        ps = torch.exp(logps)
 
        probs, classes = ps.topk(topk, dim=1)
    
        if device == "cuda":
            probs = probs.cpu()
            classes = classes.cpu()
                
    return probs.numpy().reshape(int(args['K'])), classes.numpy().reshape(int(args['K']))                

# Define classes                
def dictionaries(cat_to_name, class_to_idx):
 
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)

    idx_to_class = {}
    for key, value in class_to_idx.items():
        idx_to_class[value] = key
    
    return cat_to_name, idx_to_class

# Load checkpoint and rebuilds the model
model, class_to_idx = load_checkpoint('checkpoint.pth')

# Make prediction
probs, classes = predict(image_path, model, int(args['K']))
cat_to_name, idx_to_class = dictionaries(args['dict'], class_to_idx)

# Show class definition
print("\n Summary")

print("Probabilities and top ", args['K'], " classes")

top_class_loc = []
top_class_def =[]

for item in classes:
        top_class_loc.append(idx_to_class[item])
        top_class_def.append(cat_to_name[idx_to_class[item]])

print("\n This image is a ", top_class_def[0], " with a probability of ", probs[0],"\n")

print(probs)
print(top_class_def)