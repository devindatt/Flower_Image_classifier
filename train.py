# ------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import glob
import os
import json
import argparse

# ------------------------------------------------------------------------------- #
# Define Parsing arguments
# ------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Training neural network on dataset in directory')
parser.add_argument('-d', '--data_dir', 
					type=str, metavar='', required=True, 
					help='Directory where the image data is stored')

parser.add_argument('-m', '--model_name', 
					type=str, metavar='', required=True, 
					help='The name of the pretrained ImageNet models either densenet or vgg16')

parser.add_argument('-s', '--save_dir', 
					type=str, metavar='', required=False, 
					help='Set directory to save checkpoints')

parser.add_argument('-a', '--arch', 
					type=str, metavar='', required=False, 
					help='The model architecture, eg: vgg13, densenet121')

parser.add_argument('-l', '--learn_rate', 
					type=float, metavar='', required=False, 
					help='The model optimizers learning rate, eg: 0.01, 0.001')

parser.add_argument('-H', '--hidden_units', 
					type=int, metavar='', required=False, 
					help='The models number of nodes in hidden layer, eg: 512')

parser.add_argument('-e', '--epochs', 
					type=int, metavar='', required=False, 
					help='The number of times the whole dataset will pass thru the model for training')

parser.add_argument('-g', '--gpu', 
					type=str, metavar='', required=False, 
					help='Use when you want to run the model and tensors on a GPU for faster processing')

parser.add_argument('-k', '--top_k', 
					type=int, metavar='', required=False, 
					help='Return top K most likely classes, eg: 1 for high probability')

parser.add_argument('-c', '--cat_to_names', 
					type=str, metavar='', required=False, 
					help='The JSON file with mapping of categories to real names')

group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
group.add_argument('-v', '--verbose', action='store_true', help='print verbose')
args = parser.parse_args()



# ------------------------------------------------------------------------------- #
# Define Prepare Testing & Training Data 
# ------------------------------------------------------------------------------- #
#Function to prepare the data files
#  - Defines the transforms for the training, validation, and testing sets
#  - Create a JSON file of mapping of name categories to classes

def data_prep(data_dir, batch=64):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
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

    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch) 
  
    # Create a JSON file of mapping of name categories to classes
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    # Print out confirmation statements    
    print('Train dataset has {} batches of {} in each for a total of {} images.'.format(len(trainloader), batch, len(trainloader)*batch))
    print('Validation dataset has {} batches of {} in each for a total of {} images.'.format(len(validloader), batch, len(validloader)*batch))
    print('Test dataset has {} batches of {} in each for a total of {} images.'.format(len(testloader), batch, len(testloader)*batch))
    
    return trainloader, testloader, validloader, train_data, cat_to_name

# ------------------------------------------------------------------------------- #
# Function to Building the Model, Classifier, Criterion and Optimizer
# ------------------------------------------------------------------------------- #
def build_model(model_name='densenet121'):
        
    if model_name == 'densenet121':
    
        # Build out just the classifier architecture for Densenet121 for our flowers classes
        model = models.densenet121(pretrained=True) 
        start_features = model.classifier.in_features
        classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(start_features, 256, bias=True)),
                                    ('relu1', nn.ReLU()),
                                    ('dropout1', nn.Dropout(0.2)),

                                    ('fc2', nn.Linear(256, 256, bias=True)),
                                    ('relu2', nn.ReLU()),
                                    ('dropout2', nn.Dropout(0.1)),    

                                    ('fc4', nn.Linear(256, 102, bias=True)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))

        '''
        classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(25088, 4096, bias=True)),
                                    ('relu1', nn.ReLU()),
                                    ('dropout1', nn.Dropout(p=0.5)),
                                    ('fc2', nn.Linear(4096, 102, bias=True)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))
        '''        
    
    else:
        
        # Build out just the classifier architecture for VGG16 for our flowers classes
        model = models.vgg16(pretrained=True)
        start_features = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(start_features, 512, bias=True)),
                                  ('relu1', nn.ReLU()),
                                  ('dropout1', nn.Dropout(0.5)),

                                  ('fc2', nn.Linear(512, 512, bias=True)),
                                  ('relu2', nn.ReLU()),
                                  ('dropout2', nn.Dropout(0.5)),    

                                  ('fc4', nn.Linear(512, 102, bias=True)),

                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
        
    model.classifier = classifier

    criterion = nn.NLLLoss()
    #criterion = nn.CrossEntropyLoss()
    
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learn_rate)

    print("We are going to use the '{}' pretrained model".format(model_name))

    return model, criterion, optimizer    
# ------------------------------------------------------------------------------- #
# Function to Training the classifier
# ------------------------------------------------------------------------------- #
# Function to train and perform validation on the classifier accuracy
def train_classifier(validloader, trainloader, device, model, criterion, optimizer, epochs):
    
    steps = 0
    running_loss = 0
    print_every = 5
    model.to(device)
    print('The device is: {}'.format(device))
    for epoch in range(epochs):
#        print('In first For loop in train_classifier')
        for inputs, labels in trainloader:
#            print('In second For loop in train_classifier')
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)  # Move input & label tensors to default device
            optimizer.zero_grad()
 
            logps = model.forward(inputs)
#            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
    #                    print('The top probability here is :{}'.format(top_p))
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print("Epoch: {}/{} | ".format(epoch+1, epochs),
                      "Training loss: {:.3f} | ".format(running_loss/print_every),
                      "Testing loss: {:.3f} | ".format(test_loss/len(validloader)),
                      "Testing accuracy: {:.1f}%".format(accuracy/len(validloader)*100))

                running_loss = 0
                model.train()

# ------------------------------------------------------------------------------- #
# Function to test testdata on the classifier's accuracy
# ------------------------------------------------------------------------------- #
def testing_network(testloader, model, device):
    
    accuracy = 0
    total_images = 0
    
    model.eval()
    with torch.no_grad():

        for ii, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
#            ps = torch.exp(logps)
            
            _, predicted = torch.max(logps.data, 1)

            total_images += labels.size(0)
            accuracy += (predicted == labels).sum().item()

        print('Total correct predictions is {}'.format(accuracy))
        print('Total images in image loader was {}'.format(total_images))
            
    return (accuracy/total_images*100)

# ------------------------------------------------------------------------------- #
# Saving the Model to Checkpoint file
# ------------------------------------------------------------------------------- #
def save_model(model, optimizer, train_data):

    model.class_to_idx = train_data.class_to_idx

    #Saving the model so we can load it later for making predictions
    checkpoint = {'output_size': 102,
                  'classifier' : model.classifier,
                  'learning_rate': args.learn_rate,
                  'epochs': args.epochs,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                 }

    torch.save(checkpoint, 'checkpoint.pth')
    
    print('Model is saved to the checkpoint.pth file')
    print('\n')
    


# ------------------------------------------------------------------------------- #
#Main section
# ------------------------------------------------------------------------------- #
if __name__ == '__main__':

    if args.quiet:
        print('epochs={}, learning_rate={}, directory={}'.format(args.epochs, args.learn_rate, args.data_dir))
    elif args.verbose:
        print('The input parameters are: epochs={}, learning_rate={}, directory={}'.format(args.epochs, args.learn_rate, args.data_dir))
    else:
        print('Lets start the training!! \n')
        #Build and train your network

    # Loading the images from directory and making dataloader structures     
    trainloader, testloader, validloader, train_data, cat_to_name = data_prep(args.data_dir)   
          
    # Build and define your model
    model, criterion, optimizer = build_model(args.model_name)
    model
    
    #  Move the model if GPU processors are available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    

    model, criterion, optimizer = build_model(args.model_name)
    print("\n")
    
    train_classifier(validloader, trainloader, device, model, criterion, optimizer, args.epochs)

    print("\n")
    print("Testing accuracy: {:.2f}%".format(testing_network(testloader, model, device)))

    save_model(model, optimizer, train_data)

