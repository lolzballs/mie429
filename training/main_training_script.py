from model_loader import ModelManager 
import data 
from utils import *
import torch
import torchvision
from torch.optim import lr_scheduler
import torch.nn as nn 

import numpy as np
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
    train_loss_store,val_loss_store,train_mae_store,val_mae_store = [],[],[],[]
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    num_epochs = hp['num_epochs']
    start_epoch = 0
    checkpoint_data = {}
    if hp['resume_train']:
        start_epoch = hp['resume_epoch']
        num_epochs += hp['resume_epoch']
        best_loss = hp['resume_loss']

    for epoch in range(start_epoch,num_epochs):
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
            running_error_mean = 0
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

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                batch_mean = torch.mean(torch.abs(outputs-bone_age))
                batch_max = torch.max(torch.abs(outputs-bone_age))
                if batch_max > max_estimated_error:
                    max_estimated_error = batch_max
                # statistics
                running_loss += loss.item()
                running_error_mean += batch_mean.item()
            
            running_error_mean = running_error_mean/batch_iter
            
            if phase == 'train':
                lr_scheduler.step(running_loss)
                train_loss_store.append(running_loss)
                train_mae_store.append(running_error_mean)
            else:
                val_loss_store.append(running_loss)
                val_mae_store.append(running_error_mean)

            print(f'{phase} Total Epoch Loss: {running_loss:.3f} | Mean Average Error Across Epoch(months): {running_error_mean:.3f} | Max Regression Error Across Epoch(months): {max_estimated_error:.3f}') 

            # deep copy the model
            if phase == 'val' and running_loss < best_loss:
                best_loss = running_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Epoch Loss: {best_loss:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)

    plot_metrics = {"Train_Loss":train_loss_store, "Validation_Loss":val_loss_store, "Train_MAE":train_mae_store, "Validation_MAE":val_mae_store}
    checkpoint_data['loss'] = best_loss
    checkpoint_data['epoch'] = epoch
    return model,plot_metrics,checkpoint_data


def main():
    print("PyTorch Device:",device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='RSNA dataset location',
                        default='data')
    parser.add_argument('--result_dir', help='model weight save location',
                        default='experiment_results')
    parser.add_argument('--record_results', help='boolean toggle whether statistics of run will be tracked/written or not',
                        default=True)
    parser.add_argument('--resume_training', help='select true if this run is to resume a previously saved training and make sure to specify the PATH to saved checkpoint',
                        default=False)
    parser.add_argument('--resume_checkpoint_dir', help='PATH/directory to folder of saved checkpoint',
                        default=None)
    args = parser.parse_args()
    
    print("---Running with argparser Values:\n",args)
    #Load and compact all hyper params into dictionary
    with open("hyperparams.yaml", "r") as stream:
        try:
            hyperparams = yaml.safe_load(stream) 
            print("---Running Experiment With Hyperparams:\n",hyperparams)
        except yaml.YAMLError as exc:
            print(exc)

    if not args.resume_training:
        if args.record_results:
            currDT = datetime.now()
            experiment_id = hyperparams["experiment_name"] + "_" + currDT.strftime("%m%d_%H%M%S")
            print("---Recording experiment with id:",experiment_id)
            model_output_dir = args.result_dir + "/" + experiment_id
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)

            # Write hyperparams used back as a yaml file to experiment folder for later reference if needed 
            with open(model_output_dir+'/train_hyperparams.yaml', 'w') as outfile:
                yaml.dump(hyperparams, outfile, default_flow_style=False)
    else:
        print("---Resuming training experiment with PATH: ",args.resume_checkpoint_dir)
        experiment_id = args.resume_checkpoint_dir.split('/')[-1]
        model_output_dir = args.resume_checkpoint_dir

    train_dp, val_dp = data.RSNA(root=args.data)
    # The images from RSNA are not the same shape so we need to crop/scale/
    # post-process. For now just use a 1024x1024 crop to show how things work.
    train_dp = train_dp.map(crop_image)

    train_loader = torch.utils.data.DataLoader(dataset=train_dp, batch_size=hyperparams['batch_size'])
    val_loader = torch.utils.data.DataLoader(dataset=val_dp)
    dataloaders = {"train":train_loader, 'val':val_loader}

    modelManager = ModelManager()
    trainingModel,pretrained_transforms = modelManager.pretrained_resnet34(pretrain_source=hyperparams['pretrain_source'])

    # Modify model by adding extra fc layer to output 1 to convert model for classification to regression
    # This section will possibly need edits for different models
    default_out_features = trainingModel.fc.out_features
    trainingModel.fc = nn.Sequential(trainingModel.fc, nn.ReLU(), nn.Linear(in_features=default_out_features, out_features=1, bias=True))
    trainingModel = trainingModel.to(device)
    optimizer = torch.optim.Adam(trainingModel.parameters(),lr=hyperparams['optimizer_lr']) 
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5, patience=10,min_lr=0.001, verbose=True) #Start at large learning rate to quickly learn and only reduce when loss plateaus
    loss_function = nn.MSELoss()

    hyperparams['resume_train'] = False 
    if args.resume_training:
        checkpoint_path = model_output_dir+'/'+experiment_id+'.pt'
        checkpoint = torch.load(checkpoint_path)
        trainingModel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        hyperparams['resume_epoch'] = checkpoint['epoch']
        hyperparams['resume_loss'] = checkpoint['loss']
        hyperparams['resume_train'] = True 
        print("---Resuming Training with Prev. Best Validation Loss: {:.3f} Epoch: {}".format(checkpoint['loss'],checkpoint['epoch']))

    best_model, metrics_to_plot,checkpoint_save = train_model(trainingModel,pretrained_transforms, loss_function, optimizer, scheduler, dataloaders, hyperparams)
    
    if args.record_results:
        #save the model for future evaluation/inference only, resuming training is currently not possible, future feature to add
        torch.save({'epoch':checkpoint_save['epoch'],
                    'model_state_dict':best_model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':checkpoint_save['loss'],
                    'scheduler_state_dict':scheduler.state_dict()
                    },model_output_dir+'/'+experiment_id+'.pt')
        currDT = datetime.now()
        for i, (metric,storedata) in enumerate(metrics_to_plot.items()): # will expect train model to return a dictionary of metric name as keys and datalist as corresponding dict values for plotting
            
            plt.figure(i)
            if args.resume_training:
                plot_title = "Resumed " + experiment_id + ": " + metric+" VS Epochs"
                save_name = model_output_dir+'/resumed_'+currDT.strftime("%m%d_%H%M%S")+'_'+ experiment_id+metric+".png"
                x_vals = list(range(hyperparams['resume_epoch'],hyperparams['resume_epoch']+hyperparams['num_epochs']))
            else:
                plot_title = experiment_id + ": " + metric+" VS Epochs"
                save_name = model_output_dir+'/'+experiment_id+metric+".png"
                x_vals = list(range(0,hyperparams['num_epochs']))

            
            np_storedata = np.asarray(storedata)    #save the metric data so we can retrieve data and plot in aggregate plot later
            np.savetxt(save_name[:-4]+'.csv',np_storedata,delimiter=',')

            plt.plot(x_vals,storedata)
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.title(plot_title)
            plt.savefig(save_name, bbox_inches='tight')


    return True
    

    


if __name__ == "__main__":
    main()
