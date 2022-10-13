from model_loader import ModelManager 
import data 
from utils import *
import torch
import torchvision
from torch.optim import lr_scheduler
import torch.nn as nn 

import argparse
import yaml
import math
import time
import copy
from datetime import datetime
import os 
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model,pretrained_transforms, loss_func, optimizer, lr_scheduler, dataloaders,hp):
    since = time.time()
    loss_store, loss_per_data_store = [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    num_epochs = hp['num_epochs']
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0 # Can add more metrics if required, loss is also sufficient as accuracy for now

            # Iterate over data.
            batch_iter = 0
            for patient_id, bone_age, sex, img_batch in dataloaders[phase]:
                
                bone_age = torch.tensor(list(map(int,bone_age)))   #turn label into torch tensor
                bone_age = bone_age.unsqueeze(1)           # add extra dimension to label tensor
                
                ###
                # Preprocessing steps may require edits depending on model/experiment needs
                img_batch = img_batch.expand(-1,3,-1,-1)        #expand grayscale dim to rgb 3dim
                img_batch = pretrained_transforms(img_batch)   #apply any pretraining transforms
                ###

                img_batch = img_batch.to(device)
                bone_age = bone_age.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(img_batch)
                    loss = loss_func(outputs, bone_age)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                batch_iter += 1
                #running_avg_error += torch.sum(outputs == bone_age.data)

            if phase == 'train':
                scheduler.step()

            avg_datapoint_loss = running_loss / (hp['batch_size']*batch_iter) # calculate average loss per data sample across the epoch/dataset

            print(f'{phase} Total Epoch Loss: {running_loss:.4f} | Average Loss/Data Sample: {avg_datapoint_loss:.4f} | Max Error (months): {torch.max(outputs-bone_age):.4f}') 
            loss_store.append(running_loss)
            loss_per_data_store.append(avg_datapoint_loss)

            # deep copy the model
            if phase == 'val' and running_loss > best_loss:
                best_loss = running_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Avg Epoch Loss: {best_loss:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)

    plot_metrics = {"Total_Loss":loss_store, "Avg_LossPerDatapoint":loss_per_data_store}
    return model,plot_metrics


def main():
    print("PyTorch Device:",device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='RSNA dataset location',
                        default='data')
    parser.add_argument('--model_save_dir', help='model weight save location',
                        default='experiment_results/model_weights')
    args = parser.parse_args()
    
    
    #Load and compact all hyper params into dictionary
    with open("hyperparams.yaml", "r") as stream:
        try:
            hyperparams = yaml.safe_load(stream) 
            print("Running Experiment With Hyperparams:\n",hyperparams)
        except yaml.YAMLError as exc:
            print(exc)

    currDT = datetime.now()
    experiment_id = hyperparams["experiment_name"] + "_" + currDT.strftime("%m%d%Y_%H%M%S")
    model_output_dir = args.model_save_dir + "/" + experiment_id
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # Write hyperparams used back as a yaml file to experiment folder for later reference if needed 
    with open(model_output_dir+'/train_hyperparams.yaml', 'w') as outfile:
        yaml.dump(hyperparams, outfile, default_flow_style=False)



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
    # This section will possibly need edits for different models
    default_out_features = baseline_resnet50.fc.out_features
    baseline_resnet50.fc = nn.Sequential(baseline_resnet50.fc, nn.ReLU(), nn.Linear(in_features=default_out_features, out_features=1, bias=True))
    optimizer = torch.optim.SGD(baseline_resnet50.parameters(),lr=0.1, momentum=0.9, weight_decay=0.0001) #hyperparams taken from resnet paper
    scheduler = lr_scheduler.StepLR(optimizer,step_size = int(hyperparams['num_epochs']/4), gamma = 0.1) #decay every quarter of epochs completed following resnet convention
    loss_function = nn.MSELoss()

    best_model, metrics_to_plot = train_model(baseline_resnet50,pretrained_transforms, loss_function, optimizer, scheduler, dataloaders, hyperparams)
    torch.save(best_model,model_output_dir)

    for i, (metric,data) in enumerate(metrics_to_plot.items()): # will expect train model to return a dictionary of metric name as keys and datalist as corresponding dict values for plotting
        plt.figure(i)
        plt.plot(data)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.title(experiment_id + ": " + metric+" VS Epochs")
        plt.savefig(model_output_dir+'/'+experiment_id+metric+".png", bbox_inches='tight')


    return True
    

    


if __name__ == "__main__":
    main() #we love good coding practises
