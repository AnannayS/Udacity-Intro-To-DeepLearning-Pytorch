#  Imports

from json import load
from PIL import Image
from io import BytesIO
from charset_normalizer import models
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

#  Load in VGG19 model features

vgg = models.vgg19(pretrained=True).features

#  Freeze model parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Move model to device
vgg.to(device)

#  Load in and configure images with transforms
def load_image(img_path, max_size=400, shape=None):

        if "http" in img_path:
            response = requests.get(img_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')
    
        if max(image.size) > max_size:
            size = max_size
        else:
            size =  max(image.size)

        if shape is not None:
            size = shape

        in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))
                        ])

        image = in_transform(image)[:3,:,:].unsqueeze(0)

        return image

#  Load in sample content and style image
content = load_image('Deep Learning Course Projects\Style Transfer\pedro_pascal.jpg').to(device)

style = load_image('Deep Learning Course Projects\Style Transfer\starry_night.jpg', shape=content.shape[-2:]).to(device)

# Helper to un-normalize image and convert from Tensor to NumPy array
def im_convert(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

# # Display images
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# ax1.imshow(im_convert(content))
# ax2.imshow(im_convert(style))
# plt.show()

def get_features(image, model, layers=None):
    # Default layers are for VGGNet matching Gatys et al (2016)
    
    # Layer Mapping    
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # Content representation
                  '28': 'conv5_1'}                   

    # Fill feature dictionary
    features = {}
    x = image

    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

#  Calculate gram matrix
def gram_matrix(tensor):

    _, d, h, w = tensor.size() # Can ignore batch size
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())

    return gram 

# Get content and style images
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Calculate gram matricies for each layer
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create target image
target = content.clone().requires_grad_(True).to(device)

# Set weights for each style layer
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

# Set content vs. style weights 
content_weight = 1  # alpha
style_weight = 1e4  # beta

# # Show target image intermittently
# show_every = 400

# Iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 2000 

print("STARTING STYLE TRANSFER")
for ii in range(1, steps+1):
    
    # Get features and calculate content loss
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']**2))
    
    # Calculate style loss
    style_loss = 0
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        _, d, h, w = target_feature.shape
        
        # Target gram matrix
        target_gram = gram_matrix(target_feature)

        # Get style represenation for layer
        style_gram = style_grams[layer]

        # Calculate weighted style loss
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        
        # Add to net style loss
        style_loss += layer_style_loss / (d * h * w)
        
        
    # Sum content and style loss to get total loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # Update target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print(f'STEP {ii} DONE')
    
    # # Display intermediate images and print the loss
    # if  ii % show_every == 0:
    #     print('Total loss: ', total_loss.item())
    #     plt.imshow(im_convert(target))
    #     plt.show()

# Display content and computed target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))
plt.show() 