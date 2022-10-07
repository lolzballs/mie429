import functools
import os
import urllib.request
import zipfile
from typing import List, Optional, Tuple

import PIL
import torchdata.datapipes as dp
import torchvision.transforms.functional


URL_TRAINING = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set.zip"
URL_TRAINING_ANNOTATIONS = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set+Annotations.zip"
URL_VALIDATION = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Validation+Set.zip"


def _download_and_extract(root: str, url: str) -> str:
    name, ext = os.path.splitext(os.path.basename(url))
    path = os.path.join(root, name)

    # already extracted
    if os.path.isdir(path):
        return path

    # download if zip doesn't exist
    zipped_path = os.path.join(root, os.path.basename(url))
    if not os.path.isfile(zipped_path):
        urllib.request.urlretrieve(url, zipped_path)

    # extract
    with zipfile.ZipFile(zipped_path) as z:
        z.extractall(path)
    os.remove(zipped_path)

    return path


def _extract_validation(root: str, zip_path: str):
    # each zip contains a folder named after the zip:
    # i.e. boneage-validataion-dataset-1 inside the corresponding zip
    name, ext = os.path.splitext(os.path.basename(zip_path))

    # assume that if the zip doesn't exist, we've already extracted it
    if not os.path.isfile(zip_path):
        return

    with zipfile.ZipFile(zip_path) as z:
        for zip_info in z.infolist():
            if not zip_info.filename.startswith(f'{name}/') \
                    or zip_info.filename[-1] == '/':
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            z.extract(zip_info, path=root)

    os.remove(zip_path)


def _download_dataset(root=".data"):
    training_root = _download_and_extract(root, URL_TRAINING)
    training_root = os.path.join(training_root, 'boneage-training-dataset')

    training_annotations = _download_and_extract(root, URL_TRAINING_ANNOTATIONS)
    training_annotations = os.path.join(training_annotations, 'train.csv')

    validation_root = _download_and_extract(root, URL_VALIDATION)
    validation_root = os.path.join(validation_root, 'Bone Age Validation Set')
    # the validation zip is made up of two zips that need to be extracted
    _extract_validation(validation_root,
                        os.path.join(validation_root,
                                     'boneage-validation-dataset-1.zip'))
    _extract_validation(validation_root,
                        os.path.join(validation_root,
                                     'boneage-validation-dataset-2.zip'))

    validation_annotations = os.path.join(validation_root,
                                          'Validation Dataset.csv')

    return training_root, training_annotations, validation_root, \
        validation_annotations


def _reorder_tuple(order: Optional[List[int]], input: Tuple) -> Tuple:
    if order is None:
        return input
    return tuple(input[i] for i in order)


def _load_patient_image(training_root: str, ann: Tuple) -> Tuple:
    id, *rest = ann
    with PIL.Image.open(os.path.join(training_root, f"{id}.png")) as image:
        return id, *rest, torchvision.transforms.functional.to_tensor(image)


def _build_datapipe(root: str, annotations: str,
                    order: Optional[List[int]] = None):
    datapipe = dp.iter.IterableWrapper([annotations])
    datapipe = datapipe.open_files('b')
    datapipe = datapipe.parse_csv(skip_lines=1)
    datapipe = datapipe.map(functools.partial(_reorder_tuple, order))
    datapipe = datapipe.map(functools.partial(_load_patient_image, root))
    return datapipe


def RSNA(root=".data"):
    training_root, training_annotations, validation_root, \
        validation_annotations = _download_dataset(root)
    training_dp = _build_datapipe(training_root, training_annotations)
    validation_dp = _build_datapipe(validation_root,
                                    validation_annotations, [0, 2, 1])
    return training_dp, validation_dp
