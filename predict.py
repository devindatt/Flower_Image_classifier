# ------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------- #

from torchvision import transforms, models
from torch import nn
from collections import OrderedDict

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

# ------------------------------------------------------------------------------- #
# Define Parsing arguments
# ------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Application to make a prediction on single image')
parser.add_argument('-i', '--image_path', 
					type=str, metavar='', required=True, 
					help='Directory where the image data is stored')

parser.add_argument('-a', '--arch', 
					type=str, metavar='', required=True, 
					help='The name of the pretrained ImageNet models either densenet121 or vgg16')

parser.add_argument('-m', '--model_file', 
					type=str, metavar='', required=True, 
					help='The name of the saved checkpoint file with fully trained parameters')

group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
group.add_argument('-v', '--verbose', action='store_true', help='print verbose')
args = parser.parse_args()

# ------------------------------------------------------------------------------- #
# Rebuilding the Pretrained Model to Use to make Predictions
# ------------------------------------------------------------------------------- #
def rebuild_model(model_name ='densenet121'):
        
    if model_name == 'densenet121':
    
        # Build out just the classifier architecture for Densenet121 for our flowers classes
        model = models.densenet121(pretrained=True) 
        start_features = model.classifier.in_features
        classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(start_features, 256, bias=True)),
#                                    ('fc1', nn.Linear(1024, 256, bias=True)),
                                    ('relu1', nn.ReLU()),
                                    ('dropout1', nn.Dropout(0.2)),

                                    ('fc2', nn.Linear(256, 256, bias=True)),
                                    ('relu2', nn.ReLU()),
                                    ('dropout2', nn.Dropout(0.1)),    

                                    ('fc4', nn.Linear(256, 102, bias=True)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))

    
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
   
    return model    

# ------------------------------------------------------------------------------- #
# Loading the Model from Checkpoint file
# ------------------------------------------------------------------------------- #
def load_checkpoint(filepath):
    
    model = rebuild_model(args.arch)
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    model.classifier = checkpoint['classifier']

    return model

# ------------------------------------------------------------------------------- #
# Function process_image to simply convert a PIL image using transforms
# ------------------------------------------------------------------------------- #
def process_image(imagepath):
    
    im = Image.open(imagepath)

    # Resize, Crop, Make tensor, Normalize
    process_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    process_im = process_transform(im)
    
    arrayed_im = np.array(process_im)

    return arrayed_im

# ------------------------------------------------------------------------------- #
# Function to make a prediction with the model and print class & probabilities
# ------------------------------------------------------------------------------- #
# This function will calculate the class probabilities and display the image
def predict(image_path, model_path, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.

        image_path: string of the path to image, directly to image
        model: pytorch neural network
        top_k: integer. The top K classes to be calculated
    
        returns top_ps(k) - probability in percentages, top_classes, and flower names

    '''    
    
    model = load_checkpoint(model_path)

    #  Move the model if GPU processors are available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to('cpu')     #Can run on CPU for prediction, plus running on gpu was producing errors
#    model.to(device) 

    img = process_image(image_path)    
    torch_img = torch.from_numpy(img).type(torch.FloatTensor)
    torch_add_dim = torch_img.unsqueeze_(0)
#    torch_add_dim = torch_add_dim.to(device)
    torch_add_dim = torch_add_dim.to('cpu')
 
    # Run image thru model for prediction in evaluation mode
    model.eval()
    with torch.no_grad():
        log_ps = model.forward(torch_add_dim)

    ps = torch.exp(log_ps)
    probs_top = ps.topk(topk)[0]
    index_top = ps.topk(topk)[1]
    print('\n')
    
   # Make lists out of the probabilities and outputs
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Loadup the saved index and class list
    class_to_idx = model.class_to_idx
    
    # Inverting class-index dictionary so we can lookup by index
    indx_to_class = {x: y for y, x in class_to_idx.items()}
    
    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
            
    flower_names = []
    for classes in classes_top_list:
        flower_names.append(cat_to_name[classes])
  
    return probs_top_list, classes_top_list, flower_names
 
# ------------------------------------------------------------------------------- #
# Function to use a trained model for predictions, to make sure it makes sense
# ------------------------------------------------------------------------------- #
def sanity_check(model_file, image_path): 
    
    # Create a JSON file again of mapping of name categories to classes
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    obj_name = image_path.split('/')[2]
#    img = Image.open(image_path)
    
    flower_name = cat_to_name[obj_name].capitalize() 
 
    
    probs, classes, names = predict(image_path, model_file, cat_to_name)
    
    print('Flower Name in Image is: {}   From Category: {}'.format(flower_name, obj_name))
    print('The top 5 Category predictions are: {} '.format(classes))
    print('The top 5 Flower predictions are: {} '.format(names))
    print('The top probabilities are : {}'.format(probs*100))
   
    y_pos = np.arange(len(probs))
    
    #Local function to Printing a result banner    
    def border_msg(msg):
        row = len(msg)
        h = ''.join(['+'] + ['-' *row] + ['+'])
        result= h + '\n'"|"+msg+"|"'\n' + h
        print(result)
    
    yes_msg = '                         MATCH CONFIRMED !!!!!                                 '
    no_msg = ' XXXXXXXXXXXXXXXXXXXX   NO MATCH CONFIRMED      XXXXXXXXXXXXXXXXXXXX          '

    if classes[0] == obj_name:
        border_msg(yes_msg)
    else:
        border_msg(no_msg) 
    
# ------------------------------------------------------------------------------- #
#Main section
# ------------------------------------------------------------------------------- #
if __name__ == '__main__':

    if args.quiet:
        print('Using Image={}, and  model={}'.format(args.image_path, args.arch))
    elif args.verbose:
        print("Making a prediction using image '{}', the pretrained model is '{}', using parameters from the '{}' file".format(args.image_path, args.arch, args.model_file))
    else:
        print('Lets just get to it and make a prediction shall we??!! \n')  

        
    # Display an image along with the top 5 classes on an sample images 
    sanity_check(args.model_file, args.image_path)  



