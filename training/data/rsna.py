import functools
import hashlib
import os
import urllib.request
import zipfile
from typing import List, Optional, NamedTuple, Tuple

import PIL
import torch
import torchdata.datapipes as dp
import torchvision.transforms.functional


URL_TRAINING = (
    "https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set.zip",
    "9eab917f59bf520b4eb10803d33e7007ea3b3c2bd4df1c55aff45bc339b2736a",
)
URL_TRAINING_ANNOTATIONS = (
    "https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set+Annotations.zip",
    "f6373249b6ec85d182c62ab83f05fa76dff90ba03315e455dd6de332dffc7215",
)
URL_VALIDATION = (
    "https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Validation+Set.zip",
    "683ea5398dd9d0730af883e673176a5c6817fd35a5a59331a500fa7affcd36f0",
)
HASH_CHUNK_SIZE = 4096


class RSNAEntry(NamedTuple):
    patient_id: str
    bone_age: int
    sex: bool
    img: torch.Tensor


def _check_integrity(file: str, hash: str) -> bool:
    m = hashlib.sha256()
    with open(file, 'rb') as f:
        for byte_block in iter(lambda: f.read(HASH_CHUNK_SIZE), b''):
            m.update(byte_block)
    return m.hexdigest() == hash


def _download_and_extract(root: str, url_hash: Tuple[str, str]) -> str:
    url, hash = url_hash
    name, ext = os.path.splitext(os.path.basename(url))
    path = os.path.join(root, name)

    # already extracted
    if os.path.isdir(path):
        return path

    # download if zip doesn't exist, or if incomplete
    zipped_path = os.path.join(root, os.path.basename(url))
    exists = os.path.isfile(zipped_path) and _check_integrity(zipped_path, hash)
    if not exists:
        if os.path.isfile(zipped_path):
            print('invalid download found, re-downloading')
            os.remove(zipped_path)

        print(f'downloading {url}')
        urllib.request.urlretrieve(url, zipped_path)
    assert _check_integrity(zipped_path, hash), 'downloaded hash != expected hash'

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


def _normalize_datatypes(entry: Tuple) -> RSNAEntry:
    patient_id, bone_age, sex, img = entry
    return RSNAEntry(patient_id, int(bone_age), sex.lower() == "true", img)


def _build_datapipe(root: str, annotations: str,
                    order: Optional[List[int]] = None):
    datapipe = dp.iter.IterableWrapper([annotations])
    datapipe = datapipe.open_files('b')
    datapipe = datapipe.parse_csv(skip_lines=1)
    datapipe = datapipe.map(functools.partial(_reorder_tuple, order))
    datapipe = datapipe.map(functools.partial(_load_patient_image, root))
    datapipe = datapipe.map(_normalize_datatypes)
    return datapipe


def RSNA(root=".data") -> Tuple[dp.iter.IterDataPipe[RSNAEntry],
                                dp.iter.IterDataPipe[RSNAEntry]]:
    training_root, training_annotations, validation_root, \
        validation_annotations = _download_dataset(root)
    training_dp = _build_datapipe(training_root, training_annotations)
    validation_dp = _build_datapipe(validation_root,
                                    validation_annotations, [0, 2, 1])
    return training_dp, validation_dp
