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
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model,pretrained_transforms, loss_func, optimizer, lr_scheduler, dataloaders,hp,args,val_df):
    since = time.time()
    train_loss_store,val_loss_store,train_mae_store,val_mae_store = [],[],[],[]
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    num_epochs = hp['num_epochs']
    start_epoch = 0
    currDT = datetime.now()
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
                bone_age = bone_age.unsqueeze(1)           # add extra dimension to label tensor
                patient_id = np.array(list(map(int, patient_id))) #convert from tuple of stringed ints to list of ints
                data_size_count += img_batch.shape[0]
                ###
                # Preprocessing steps may require edits depending on model/experiment needs
                img_batch = img_batch.expand(-1,1,-1,-1)        #expand grayscale dim to rgb 3dim
                # pretrained transforms not used when self-defined transforms are built into dataloader
                # img_batch = pretrained_transforms(img_batch)   #apply any pretraining transforms

                ###

                img_batch = img_batch.to(device)
                bone_age = bone_age.to(device)
                sex = sex.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if model.__class__.__name__ == "Bilbily":
                        outputs = model(img_batch, sex.float())
                    else:
                        outputs = model(img_batch)
                    loss = loss_func(outputs, bone_age.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                abs_prediction_errors = torch.abs(outputs-bone_age)
                batch_mean = torch.mean(abs_prediction_errors)
                batch_max = torch.max(abs_prediction_errors)
                if batch_max > max_estimated_error:
                    max_estimated_error = batch_max
                # statistics
                if phase == 'val': #store val performance histogram stats. note during validation batch size is 1
                    if abs_prediction_errors.item() > 200:
                        val_df.loc[patient_id[0],'error_beyond_200'] += 1
                    elif abs_prediction_errors.item() > 100:
                        val_df.loc[patient_id[0],'error_beyond_100'] += 1
                    elif abs_prediction_errors.item() > 40:
                        val_df.loc[patient_id[0],'error_beyond_40'] += 1
                    elif abs_prediction_errors.item() > 20:
                        val_df.loc[patient_id[0],'error_beyond_20'] += 1
                    elif abs_prediction_errors.item() > 10:
                        val_df.loc[patient_id[0],'error_beyond_10'] += 1
                    else:
                        val_df.loc[patient_id[0],'error_less_10'] += 1

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
        
        #save a general checkpoint every hp['saving_frequency'] epochs
        if epoch > 0 and epoch%hp['saving_frequency'] == 0:
            print("---Saving General Checkpoint")
            plot_metrics = {"Train_Loss":train_loss_store, "Validation_Loss":val_loss_store, "Train_MAE":train_mae_store, "Validation_MAE":val_mae_store}
            torch.save({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':running_loss,
                        'scheduler_state_dict':lr_scheduler.state_dict()
                        },hp['model_output_dir']+'/'+hp['experiment_id']+'.pt')
            #Save final best model's weights for inference only
            print("---Saving Best Model Weights")
            torch.save(best_model_wts,hp['model_output_dir']+'/'+hp['experiment_id']+'_best_model.pt')

            for i, (metric,storedata) in enumerate(plot_metrics.items()): # will expect train model to return a dictionary of metric name as keys and datalist as corresponding dict values for plotting
                
                plt.figure(i)
                if args.resume_training:
                    plot_title = "Resumed " + hp['experiment_id'] + ": " + metric+" VS Epochs"
                    save_name = hp['model_output_dir']+'/resumed_'+currDT.strftime("%m%d_%H%M%S")+'_'+ hp['experiment_id']+metric+".png"
                    x_vals = list(range(hp['resume_epoch'],hp['resume_epoch']+len(storedata)))
                else:
                    plot_title = hp['experiment_id'] + ": " + metric+" VS Epochs"
                    save_name = hp['model_output_dir']+'/'+hp['experiment_id']+metric+".png"
                    x_vals = list(range(0,len(storedata)))

                
                np_storedata = np.asarray(storedata)    #save the metric data so we can retrieve data and plot in aggregate plot later
                np.savetxt(save_name[:-4]+'.csv',np_storedata,delimiter=',')
                plt.plot(x_vals,storedata)
                plt.xlabel("Epochs")
                plt.ylabel(metric)
                plt.title(plot_title)
                plt.savefig(save_name, bbox_inches='tight')

            #Save dataframe data to csv
            val_df.to_csv(hp['model_output_dir']+'/'+hp['experiment_id']+"_error_histo_data.csv")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Epoch Loss: {best_loss:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


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

    model_params = copy.deepcopy(hyperparams['model'])
    del model_params['name']

    modelManager = ModelManager()


    train_transforms = modelManager.get_data_transform(hyperparams['train_image_transforms'])
    val_transforms = modelManager.get_data_transform(hyperparams['val_image_transforms'])

    train_dp, val_dp = data.RSNA(root=args.data)
    train_dp = train_dp.map(apply_to_image(train_transforms))
    val_dp = val_dp.map(apply_to_image(val_transforms))
    train_loader = torch.utils.data.DataLoader(dataset=train_dp, batch_size=hyperparams['batch_size'])
    val_loader = torch.utils.data.DataLoader(dataset=val_dp)
    dataloaders = {"train":train_loader, 'val':val_loader}

    trainingModel, pretrained_transforms = modelManager.get_model( #Pretrained transforms not used if transforms are self defined above
        model_name=hyperparams['model']['name'],
        pretrain_source=hyperparams['pretrain_source'],
        **model_params,
    )
    # Modify model by adding extra fc layer to output 1 to convert model for classification to regression
    # This section will possibly need edits for different models
    #TODO: currently assumes last layer is named "fc", fix in future
    # default_out_features = trainingModel.fc.out_features
    # trainingModel.fc = nn.Sequential(trainingModel.fc, nn.ReLU(), nn.Linear(in_features=default_out_features, out_features=1, bias=True))
    trainingModel = trainingModel.to(device)
    optimizer = torch.optim.AdamW(trainingModel.parameters(),lr=hyperparams['optimizer_lr']) 
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=hyperparams['scheduler_params']['factor'], 
                                                                    patience=hyperparams['scheduler_params']['patience'],
                                                                    min_lr=hyperparams['scheduler_params']['min_lr'], 
                                                                    cooldown=hyperparams['scheduler_params']['cooldown'],
                                                                    verbose=True) #Start at large learning rate to quickly learn and only reduce when loss plateaus

    loss_function = nn.MSELoss()

    hyperparams['resume_train'] = False 
    hyperparams['experiment_id'] = experiment_id
    hyperparams['model_output_dir'] = model_output_dir
    if args.resume_training:
        checkpoint_path = model_output_dir+'/'+experiment_id+'.pt'
        checkpoint = torch.load(checkpoint_path)
        trainingModel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        hyperparams['resume_epoch'] = checkpoint['epoch']
        hyperparams['resume_loss'] = checkpoint['loss']
        hyperparams['resume_train'] = True 
        print("---Resuming Training with Prev. Most Recently Saved Validation Loss: {:.3f} Epoch: {}".format(checkpoint['loss'],checkpoint['epoch']))

    # Setup pandas dataframe to store validation performance stats 
    val_df = pd.read_csv(args.data+'/'+'Bone+Age+Validation+Set/Bone Age Validation Set/Validation Dataset.csv',index_col='Image ID')
    #initialize new columns of 0s to track validation prediction performance histogram data
    val_df['error_beyond_10'] = 0 
    val_df['error_beyond_20'] = 0
    val_df['error_beyond_40'] = 0
    val_df['error_beyond_100'] = 0
    val_df['error_beyond_200'] = 0
    val_df['error_less_10'] = 0

    best_model = train_model(trainingModel,pretrained_transforms, loss_function, optimizer, scheduler, dataloaders, hyperparams,args,val_df)

    #Save final best model's weights for inference only
    print("---Saving Best Model Weights")
    torch.save(best_model.state_dict(),model_output_dir+'/'+experiment_id+'_best_model.pt')

    return True
    

    


if __name__ == "__main__":
    main()
