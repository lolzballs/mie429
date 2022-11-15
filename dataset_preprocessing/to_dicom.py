import argparse
import os

from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
import pydicom.uid
from tqdm import tqdm


def generate_dataset(image_folder: str, patient_id: str, sex: bool,
                     bone_age: str) -> Dataset:
    sop_uid = pydicom.uid.generate_uid()
    study_uid = pydicom.uid.generate_uid()
    series_uid = pydicom.uid.generate_uid()

    with Image.open(os.path.join(image_folder, f'{patient_id}.png')) as im:
        width, height, pixel_data = im.width, im.height, im.tobytes('raw')

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.ComputedRadiographyImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = Dataset()
    ds.SOPClassUID = pydicom.uid.ComputedRadiographyImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyDate = '20000101'
    ds.StudyDescription = 'XR BONE AGE LEFT 1 VIEW'
    ds.SeriesDate = '20000101'
    ds.SpecificCharacterSet = 'ISO_IR 100'

    ds.Modality = 'CR'
    ds.FilterType = 'none'
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid

    ds.PatientName = f'RSNA {patient_id}'
    ds.PatientID = f'RSNA-{patient_id}'
    ds.PatientSex = 'M' if sex else 'F'
    ds.ImageComments = bone_age
    ds.StudyID = ''
    ds.SeriesNumber = '1'
    ds.InstanceNumber = '1'

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows = height
    ds.Columns = width
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_data

    ds.file_meta = file_meta
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generates dicom')
    parser.add_argument('image_dir')
    parser.add_argument('csv')
    parser.add_argument('out_dir')
    args = parser.parse_args()

    with open(args.csv, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            if i == 0:
                assert line.strip() == "Image ID,male,Bone Age (months)",\
                    "must be run on validation dataset"
                continue
            patient_id, male, bone_age = line.split(',')
            ds = generate_dataset(args.image_dir, patient_id,
                                  male.lower() == 'true', bone_age)
            ds.save_as(os.path.join(args.out_dir, f'{patient_id}.dcm'), False)
