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
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model,pretrained_transforms, loss_func, optimizer, lr_scheduler, dataloaders,hp):
    since = time.time()
    train_loss_store,val_loss_store = [],[]
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    num_epochs = hp['num_epochs']
    if torch.cuda.is_available():
        model = model.cuda()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print("Training Phase")
            else:
                model.eval()   # Set model to evaluate mode
                print("Validation Phase")

            running_loss = 0.0 # Can add more metrics if required, loss is also sufficient as accuracy for now

            # Iterate over data.
            data_size_count = 0 #variable to count data samples can use if needed
            max_estimated_error = 0
            for batch_iter, (patient_id, bone_age, sex, img_batch) in enumerate(tqdm(dataloaders[phase])):
                
                bone_age = torch.tensor(list(map(int,bone_age)))  #turn label into torch tensor
                bone_age = bone_age.unsqueeze(1)           # add extra dimension to label tensor
                
                data_size_count += img_batch.shape[0]
                
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
                    loss = loss_func(outputs, bone_age.float())

                    batch_max = torch.max(torch.abs(outputs-bone_age))
                    if batch_max > max_estimated_error:
                        max_estimated_error = batch_max

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            if phase == 'train':
                lr_scheduler.step(running_loss)
                train_loss_store.append(running_loss)
            else:
                val_loss_store.append(running_loss)


            print(f'{phase} Total Epoch Loss: {running_loss:.4f} | Max Regression Error Across Epoch(months): {max_estimated_error:.4f}') 

            # deep copy the model
            if phase == 'val' and running_loss < best_loss:
                best_loss = running_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Epoch Loss: {best_loss:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)

    plot_metrics = {"Train_Loss":train_loss_store, "Validation_Loss":val_loss_store}
    return model,plot_metrics


def main():
    print("PyTorch Device:",device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='RSNA dataset location',
                        default='data')
    parser.add_argument('--result_dir', help='model weight save location',
                        default='experiment_results')
    parser.add_argument('--record_results', help='boolean toggle whether statistics of run will be tracked/written or not',
                        default=True)
    args = parser.parse_args()
    
    print("Running with argparser Values:\n",args)
    #Load and compact all hyper params into dictionary
    with open("hyperparams.yaml", "r") as stream:
        try:
            hyperparams = yaml.safe_load(stream) 
            print("Running Experiment With Hyperparams:\n",hyperparams)
        except yaml.YAMLError as exc:
            print(exc)

    if args.record_results:
        currDT = datetime.now()
        experiment_id = hyperparams["experiment_name"] + "_" + currDT.strftime("%m%d%Y_%H%M%S")
        print("Recording experiment with id:",experiment_id)
        model_output_dir = args.result_dir + "/" + experiment_id
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
    trainingModel,pretrained_transforms = modelManager.get_model(model_name=hyperparams['model_name'], pretrain_source=hyperparams['pretrain_source'])
    # Modify model by adding extra fc layer to output 1 to convert model for classification to regression
    # This section will possibly need edits for different models
    default_out_features = trainingModel.fc.out_features
    trainingModel.fc = nn.Sequential(trainingModel.fc, nn.ReLU(), nn.Linear(in_features=default_out_features, out_features=1, bias=True))
    optimizer = torch.optim.Adam(trainingModel.parameters(),lr=hyperparams['optimizer_lr']) 
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1, patience=5, verbose=True) #Start at large learning rate to quickly learn and only reduce when loss plateaus
    loss_function = nn.MSELoss()

    best_model, metrics_to_plot = train_model(trainingModel,pretrained_transforms, loss_function, optimizer, scheduler, dataloaders, hyperparams)
    
    #save the model for future evaluation/inference only, resuming training is currently not possible, future feature to add
    torch.save(best_model,model_output_dir+'/'+experiment_id+'.pt')

    if args.record_results:
        for i, (metric,storedata) in enumerate(metrics_to_plot.items()): # will expect train model to return a dictionary of metric name as keys and datalist as corresponding dict values for plotting
            plt.figure(i)
            plt.plot(storedata)
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.title(experiment_id + ": " + metric+" VS Epochs")
            plt.savefig(model_output_dir+'/'+experiment_id+metric+".png", bbox_inches='tight')


    return True
    

    


if __name__ == "__main__":
    main() #we love good coding practises
