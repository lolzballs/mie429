### Convert DICOM --> PNG; to use with THP data (not training)
# sample image is a chest X-ray from https://github.com/ImagingInformatics/hackathon-images/blob/master/Ravi%20SIIM/W_Chest_PA_3172/IM-0031-0001.dcm

import numpy as np
import pandas as pd
import pydicom
import os
from PIL import Image


def get_image(file_name, ds, dest_dir = '/pngs'): 
    img = ds.pixel_array # get image array
    img = np.array(img, dtype = float) 
    img = (img - img.min()) / (img.max() - img.min()) * 255.0 # normalize values
    img = img.astype(np.uint8)

    im = Image.fromarray(img)
    img_path = os.getcwd() + dest_dir + '/' + file_name.replace('.dcm','.png')
    im.save(img_path)

    return

def get_patient_info(ds, all_patient_dict):
    all_patient_dict['si_uid'].append(str(ds.SeriesInstanceUID)) 
    all_patient_dict['patient_sex'].append(ds.PatientSex)
    # all_patient_dict['img_orient'] = list(ds.ImageOrientationPatient) # I think this just gets axial, coronal, etc. orientations
    return all_patient_dict

def convert(src_dir = '/dcms', dest_dir = '/pngs'): # specify file paths
    src = os.getcwd() + src_dir
    
    patients_info = {'si_uid': [], 'patient_sex': []}
    for file in os.listdir(src):
        if file.endswith(".dcm"):
            file_path = src + '/' + file 
            ds = pydicom.read_file(file_path) # FileDataset
            get_image(file, ds, dest_dir)
            patients_info = get_patient_info(ds, patients_info)
    return patients_info

if __name__ == '__main__':
    result = convert()
    df = pd.DataFrame.from_dict(result)
    df = df.set_index('si_uid')
    df.to_csv('patient_info.csv')