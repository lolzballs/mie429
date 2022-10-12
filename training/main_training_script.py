from model_loader import ModelManager 
import data 

import torch
import torchvision
from torch.optim import lr_scheduler
import torch.nn as nn 

import argparse
import yaml
import math
import time
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def crop_image(entry):
    *rest, image = entry
    return *rest, torchvision.transforms.functional.center_crop(image, (1024, 1024))


def train_model(model,pretrained_transforms, loss_func, optimizer, lr_scheduler, num_epochs, dataloaders):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0
    print("print check: dataloader size:",len(dataloaders['train'].dataset))

    quit()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            #running_avg_error = 0 no need for accuracy because due to regression, MSE loss can be monitored as our error measure/accuracy

            # Iterate over data.
            batch_iter = 0
            for patient_id, bone_age, sex, img_batch in dataloaders[phase]:
                
                bone_age = torch.tensor(list(map(int,bone_age)))   #turn label into torch tensor
                bone_age = bone_age.unsqueeze(1)           # add extra dimension to label tensor
                img_batch = img_batch.expand(-1,3,-1,-1) #expand grayscale dim to rgb 3dim
                img_batch = pretrained_transforms(img_batch)   #apply resnet50 pretrained transforms
                
                img_batch = img_batch.to(device)
                bone_age = bone_age.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(img_batch)

                    print(f"print check: difference:{outputs-bone_age}")
                    
                    loss = loss_func(outputs, bone_age)

                    print(f"print check: loss:{loss.item()}")

                    quit()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() #* img_batch.size(0)
                batch_iter += 1
                #running_avg_error += torch.sum(outputs == bone_age.data)

            if phase == 'train':
                scheduler.step()

            avg_epoch_loss = running_loss / batch_iter # calculate average loss per batch across the epoch
            #epoch_acc = running_avg_error.double() / dataset_sizes[phase]

            print(f'{phase} Average Epoch Loss: {avg_epoch_loss:.4f}')

            # deep copy the model
            if phase == 'val' and avg_epoch_loss > best_loss:
                best_loss = avg_epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Avg Epoch Loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    print("PyTorch Device:",device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='RSNA dataset location',
                        default='data')
    args = parser.parse_args()

    #Load and compact all hyper params into dictionary
    with open("hyperparams.yaml", "r") as stream:
        try:
            hyperparams = yaml.safe_load(stream) 
        except yaml.YAMLError as exc:
            print(exc)

    train_dp, val_dp = data.RSNA(root=args.data)
    # The images from RSNA are not the same shape so we need to crop/scale/
    # post-process. For now just use a 1024x1024 crop to show how things work.
    train_dp = train_dp.map(crop_image)

    train_loader = torch.utils.data.DataLoader(dataset=train_dp, batch_size=hyperparams['batch_size'])
    val_loader = torch.utils.data.DataLoader(dataset=val_dp)
    dataloaders = {"train":train_loader, 'val':val_loader}

    modelManager = ModelManager()
    baseline_resnet50,pretrained_transforms = modelManager.pretrained_resnet50(pretrain_source=hyperparams['pretrain_source'])
    # Modify model by adding extra fc layer to output 1 to convert model for classification to regression
    default_out_features = baseline_resnet50.fc.out_features
    baseline_resnet50.fc = nn.Sequential(baseline_resnet50.fc, nn.ReLU(), nn.Linear(in_features=default_out_features, out_features=1, bias=True))

    optimizer = torch.optim.SGD(baseline_resnet50.parameters(),lr=0.1, momentum=0.9, weight_decay=0.0001) #hyperparams taken from resnet paper
    scheduler = lr_scheduler.StepLR(optimizer,step_size = int(hyperparams['num_epochs']/4), gamma = 0.1) #decay every quarter of epochs completed following resnet convention
    loss_function = nn.MSELoss()

    best_model = train_model(baseline_resnet50,pretrained_transforms, loss_function, optimizer, scheduler, hyperparams['num_epochs'], dataloaders)

    # No model saving functionality implemented yet
    '''
    Random garbage testing forloop
    for patient_id, bone_age, sex, image in train_loader:
        print(bone_age)
        #print(f"patient_id datatype:{type(patient_id)}, bone_age type: {type(bone_age)}, image type:{type(image)}")
        image = image.expand(-1,3,-1,-1)
        transformed_imgs_batch = pretrained_transforms(image)
        print(f"Model FC Summary:{baseline_resnet50.fc}")
        baseline_resnet50.fc = nn.Sequential(baseline_resnet50.fc, nn.ReLU(), nn.Linear(in_features=baseline_resnet50.fc.out_features, out_features=1, bias=True))
        print(f"Post addition Model FC Summary:{baseline_resnet50.fc}")
        print(f"Model Summary:{baseline_resnet50}")
        quit()
    '''

    return True
    

    


if __name__ == "__main__":
    main() #we love good coding practises
