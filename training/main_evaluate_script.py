from model_loader import ModelManager 
import data 
from utils import *
import torch
import torchvision
from torch.optim import lr_scheduler
import torch.nn as nn 

import numpy as np
import pandas as pd
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

def main():

    print("PyTorch Device:",device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='RSNA dataset location',
                        default='data') #NOTE: Remember to define the directory containing feature removed data for best results
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
    
    ### NOTE: define evaluation data transforms in hyperparam.yaml val_image_transforms, train image transform list is not relevant at all in this eval script
    ### Model params in hyperparam.yaml should match loaded model params, other params are irrelevant
    with open("hyperparams.yaml", "r") as stream:
        try:
            hyperparams = yaml.safe_load(stream) 
            print("---Running Experiment With Hyperparams:\n",hyperparams)
        except yaml.YAMLError as exc:
            print(exc)

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

    evalModel, pretrained_transforms = modelManager.get_model( #Pretrained transforms not used if transforms are self defined above
        model_name=hyperparams['model']['name'],
        pretrain_source=hyperparams['pretrain_source'],
        **model_params,
    )

    # remove map_location cpu argument if evaluating on GPU, else keep it if evaluating on cpu
    evalModel.load_state_dict(torch.load('experiment_results/bilbily_jayce_setup_v2_1122_222053_vMAE6_90/bilbily_jayce_setup_v2_1122_222053_best_model.pt', map_location=torch.device('cpu')))
    evalModel.eval()
    predictions,truth,patientid = [],[],[]

    for batch_iter, (patient_id, bone_age, sex, img_batch) in enumerate(tqdm(dataloaders['val'])):
        bone_age = bone_age.unsqueeze(1)           # add extra dimension to label tensor
        patient_id = np.array(list(map(int, patient_id))) #convert from tuple of stringed ints to list of ints
        print("Evaluating id:",patient_id[0])
        img_batch = img_batch.expand(-1,1,-1,-1)        #expand grayscale dim to rgb 3dim
        img_batch = img_batch.to(device)
        bone_age = bone_age.to(device)
        sex = sex.to(device)
        outputs = evalModel(img_batch, sex.float())
        print("Prediction:{} | Ground Truth Age:{}\n".format(outputs.item(),bone_age.item()))
        predictions.append(outputs.item())
        truth.append(bone_age.item())
        patientid.append(patient_id[0])

    predictions = np.array(predictions)
    truth = np.array(truth)
    patientid = np.array(patientid)
    df = pd.DataFrame({'patient_id':patientid, 'prediction_boneage':predictions, 'ground_truth_boneage':truth})

    ### NOTE: Define output csv name here
    df.to_csv('bilbily8_08_eval_results.csv',index=False)
    print("Average vMAE across epoch:{}".format(np.mean(abs(np.subtract(predictions,truth)))))

    return True


if __name__ == "__main__":
    main()