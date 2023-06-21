#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torch.nn.functional as F
# Transforms are common image transformations. They can be chained together using Sequential. This allows you to modify your image
import torchvision.transforms as transforms
# Torchvision provides many built-in datasets in the torchvision.datasets module, as well as utility classes for building your own datasets.
import torchvision.datasets as datasets
import json
import os
import argparse
import logging
import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Configuration for Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion):
    #logger.info(f"Epoch: {epoch_no} - Testing Model on Complete Testing Dataset!")
    model.eval()
    #hook.set_mode(smd.modes.EVAL) # set debugger hook mode to EVAL
    running_loss=0
    correct=0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    


    
    
#---------------------------------------------------------------------------------------------
def train(model, train_loader, criterion, optimizer, device, test_loader):
    #logger.info(f"Epoch: {epoch_no} - Training Model on Complete Training Dataset!")
    model.train()
    #hook.set_mode(smd.modes.TRAIN) # set debugger hook mode to TRAIN
    epochs=2
    best_loss=1e6
    #image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(train_loader[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)

                with torch.no_grad():
                    running_corrects += torch.sum(preds == labels).item()

                if phase=='train':
                    loss = F.nll_loss(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                #NOTE: Comment lines below to train and test on whole dataset
                if running_samples>(0.2*len(train_loader[phase].dataset)):
                    break

            epoch_loss = running_loss / len(train_loader[phase].dataset)
            epoch_acc = running_corrects / len(train_loader[phase].dataset)
        
            print(f'Epoch : {epoch}-{phase}, epoch loss = {epoch_loss}, epoch_acc = {epoch_acc}')

            test(model, test_loader, criterion)
    
            '''
            TODO: Save the trained model
            '''
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), path)


    return model
#---------------------------------------------------------------------------------------------

def create_model():
    # importing pre-trained model
    model = models.resnet18(pretrained=True)
    
    #we need to freeze all the convolutional layers which we do by their requires_grad() attribute to False. 
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    
    # number of features is the output from the imported model and 133 is the number of classes in the training set
    model.fc = nn.Sequential(nn.Linear(num_features, 5))

    return model


#---------------------------------------------------------------------------------------------
def create_data_loaders(data , batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # creating the dataloaders
    dataloaders = {
        split : torch.utils.data.DataLoader(data[split], batch_size, shuffle=True)
        for split in ['train', 'valid', 'test']
    }

    return dataloaders
    
    
#---------------------------------------------------------------------------------------------
def main(args):
    # logging the used hyperparameters
    #print(f"Hyperparameters selected : /n#epochs : {args.epochs}\nBatch Size : {args.batch_size}\nLearning Rate : {args.lr}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=create_model()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # loading the data
    # declaring the data_transforms for train, validation and test datasets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # loading the datasets
    image_datasets = {
        split : datasets.ImageFolder(os.path.join(args.data_dir, split), data_transforms[split])
        for split in ['train', 'valid', 'test']
    }

    dataloaders = create_data_loaders(image_datasets , args.batch_size)
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    test_loader = dataloaders['test']

    train_loaders = {
        'train' : train_loader,
        'valid' : valid_loader
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=create_model()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), args.lr)
    model=train(model, train_loaders, loss_criterion, optimizer, device, test_loader)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    #test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    #path = os.path.join(args.model_dir, 'model.pth')
    #torch.save(model.state_dict(), path)







#---------------------------------------------------------------------------------------------
if __name__=='__main__':

    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument( "--lr", type = float, default = 0.1, metavar = "N", help = "learning rate (default: 0.1)" )

    parser.add_argument( "--batch_size", type = int, default = 64, metavar = "N", help = "input batch size for training (default: 64)" )
    

    # Container environment
    #parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    #parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    
    # When the training job finishes, the container and its file system will be deleted, with the exception of the /opt/ml/model and /opt/ml/output directories. Use /opt/ml/model to save the model checkpoints.
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    #parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()
    
    
    main(args)
