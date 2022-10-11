from model_loader import ModelManager 
import data 
import torch
import torchvision
import argparse


def crop_image(entry):
    *rest, image = entry
    return *rest, torchvision.transforms.functional.center_crop(image, (1024, 1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='RSNA dataset location',
                        default='/shared/data')
    args = parser.parse_args()

    train_dp, val_dp = data.RSNA(root=args.data)
    # The images from RSNA are not the same shape so we need to crop/scale/
    # post-process. For now just use a 1024x1024 crop to show how things work.
    train_dp = train_dp.map(crop_image)

    train_loader = torch.utils.data.DataLoader(dataset=train_dp, batch_size=32)
    val_loader = torch.utils.data.DataLoader(dataset=val_dp)

    baseline_resnet50,pretrained_transforms = ModelManager.pretrained_resnet50(pretrain_source="imagenet")
    for patient_id, bone_age, sex, image in train_loader:
        print(patient_id, bone_age, sex)
        print("data_img shape:",image.shape)
