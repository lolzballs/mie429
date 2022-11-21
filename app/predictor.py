import copy
import dataclasses
import queue
import os
import threading
import sys
from typing import Optional, Tuple

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pydicom.dataset import Dataset, FileMetaDataset
import pydicom.uid
import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..',
                             'dataset_preprocessing'))
from image_matcher import ImageMatcher

from model_loader import ModelManager
from models import Bilbily

# Our UID was obtained from Medical Connections
# Suballocated as follows:
#   1. Series
#   2. SOP
# Each of the above are further suballocated as:       
#   2. Prediction + Reference
#   3. Heatmap
#   4. Growth Chart
# Example: '1.2.826.0.1.3680043.10.1082.1.4.' is the prefix for all series of
#          growth charts.
PREDICTOR_UID_ROOT = '1.2.826.0.1.3680043.10.1082.'



@dataclasses.dataclass(frozen=True)
class Prediction:
    prediction: Dataset
    heatmap: Dataset
    growth_chart: Dataset

class ModelWithTransforms(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        model_manager = ModelManager()
        self.transforms = torch.nn.Sequential(*model_manager.get_data_transform([
            'resize',
            'normalize',
        ]).transforms)

    def forward(self, x: torch.Tensor, sex: torch.Tensor):
        with torch.no_grad():
            x = self.transforms(x)
            x = torchvision.transforms.functional.adjust_contrast(x, contrast_factor=1.2)
        return self.model(x, sex)

### for heatmap
class SaveFeatures():
    """ Extract pretrained activations"""
    features=None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()

class Predictor:
    _process_queue: queue.Queue[Dataset]
    _result_queue: queue.Queue[Prediction]
    _model: torch.nn.Module

    def __init__(self, model_path: str, icon_path: str, num_workers: int):
        self._image_matcher = ImageMatcher(cv2.imread(icon_path))
        # self._model = torch.jit.load(model_path)
        # self._model.eval()
        # print(self._model)

        self._model_load = Bilbily()
        self._model_load.load_state_dict(torch.load('bilbily.pt', map_location=torch.device('cpu'))['model_state_dict'])
        self._model = ModelWithTransforms(self._model_load)
        self._model.eval()

        self._running = True
        self._process_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._font = ImageFont.load_default()
        self._workers = [_PredictorWorker(self, self._process_queue,
                                          self._result_queue)
                         for _ in range(num_workers)]

        for worker in self._workers:
            worker.start()

    def predict(self, dataset: Dataset):
        self._process_queue.put(dataset)

    def get_result(self):
        item = self._result_queue.get()
        self._result_queue.task_done()
        return item

    def stop(self):
        self._running = False

        self._process_queue.join()
        for worker in self._workers:
            worker.join()

        self._result_queue.join()

    def _run_model(self, image: np.ndarray, sex: int) \
            -> Tuple[float, np.ndarray]:
        # remove signatures
        self._image_matcher.upload_search_image(image)
        _, result_img = self._image_matcher.find_object(padding_X=125,
                                                        padding_Y=175,
                                                        draw_box=False)
        if result_img is not None:
            image = result_img

        image = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)

        # for XAI
        # final_layer = self._model._modules['model']._modules['inception']._modules['Mixed_7c']
        final_layer = self._model._modules.get('model')._modules.get('inception')._modules.get('Mixed_7c')
        activated_features = SaveFeatures(final_layer)

        age = self._model(image, torch.tensor(sex).unsqueeze(0))
        activated_features.remove() #let's do this to be safe
        
        # TODO: XAI stuff
        # heatmap = np.random.random_integers(128, 256, image.shape)
        heatmap = self._get_heatmap(image, activated_features)
        return age, heatmap

    def _get_heatmap(self, image, activated_features) -> np.ndarray:
        weight_softmax_params = list(self._model._modules.get('model')._modules['fc'].parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        final_fc_weights = weight_softmax_params[2].data.numpy()

        image = image.expand(-1,3,-1,-1).squeeze().numpy().transpose((1, 2, 0))
        image = image * 255
        heatmap_only = self._get_CAM(activated_features.features, weight_softmax, final_fc_weights)
        heatmap = cv2.resize(heatmap_only, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + image * 0.6
        superimposed_img = superimposed_img.astype(np.uint8)
        ### testing purposes
        im = Image.fromarray(superimposed_img)
        im.save("heatmap.png")
        return superimposed_img
    
    def _get_CAM(self, feature_conv, weight_fc, final_fc_weights) -> np.ndarray: # returns heatmap
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[:,None,:nc].dot(feature_conv.reshape((-1, nc, h*w))).squeeze()
        cam = cam.T@final_fc_weights.T
        cam = cam.reshape(h, w)
        
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return cam_img

class _PredictorWorker(threading.Thread):
    def __init__(self, predictor: Predictor, in_queue: queue.Queue[Dataset],
                 out_queue: queue.Queue[Prediction]):
        threading.Thread.__init__(self, name='PredictorWorker')

        self._predictor = predictor
        self._in_queue = in_queue
        self._out_queue = out_queue

    def run(self):
        while self._predictor._running:
            try:
                item = self._in_queue.get(timeout=1)
            except queue.Empty:
                continue

            result = self._process_item(item)
            if result is not None:
                self._out_queue.put(result)
            self._in_queue.task_done()

    def _process_item(self, item: Dataset) -> Optional[Prediction]:
        if item.PhotometricInterpretation != 'MONOCHROME2':
            print(f'warning: {item.StudyInstanceUID} had unsupported '
                  'photometric interpretation {item.PhotometricInterpretation} '
                  'and was skipped')
            return

        item.decompress()
        prediction, heatmap = self._predictor._run_model(
            item.pixel_array,
            1 if item.PatientSex == 'M' else 0,
        )

        pred_int = torch.round(prediction).int().item()

        image = Image.fromarray(item.pixel_array, 'L')
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), f'{pred_int}', fill="#fff",
                  font=self._predictor._font, stroke_width=5, stroke_fill="#000")

        atlas_dir = 'training/data/male_atlas' if item.PatientSex == 'M' else 'training/data/female_atlas'
        atlas_imgs = os.listdir(atlas_dir)
        atlas_nums = np.array([int(x.split('.')[0]) for x in atlas_imgs if x.split('.')[1] == 'png'])
        atlas_nums.sort()
        closest_idx = (np.abs(atlas_nums - pred_int)).argmin() #https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        closest_val = atlas_nums[closest_idx]
        if closest_val > pred_int:
            more_idx = closest_idx
            less_idx = closest_idx - 1
        else:
            more_idx = closest_idx + 1
            less_idx = closest_idx
        less_val = atlas_nums[less_idx]
        less_img = Image.open(atlas_dir + "/" + str(less_val) + '.png')
        less_img = less_img.resize((image.size[1], image.size[1]))
        more_val = atlas_nums[more_idx]
        more_img = Image.open(atlas_dir + "/" + str(more_val) + '.png')
        more_img = more_img.resize((image.size[1], image.size[1]))

        images = [less_img, image, more_img] #https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        image_w_atlas = Image.new('L', (total_width, max_height))

        x_offset = 0
        for im in images:
            image_w_atlas.paste(im, (x_offset,0))
            x_offset += im.size[0]

        image_w_atlas = np.array(image_w_atlas)
        image = np.array(image)

        # quick google search shows that chronological age in years- we can check this later
        # female chart goes up to 16
        chronological_age = [3/12, 6/12, 9/12, 12/12, 18/12, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        if item.PatientSex == 'M':
            chronological_age.append(17)
            # bone age in months according to graph??
            bone_age = [3.01, 6.09, 9.56, 12.74, 19.36, 25.97, 32.40, 38.21, 43.89, 49.04, 56.00, 62.43, 75.46, 
                        88.20, 101.38, 113.90, 125.68, 137.32, 148.82, 158.39, 170.02, 182.72, 195.32, 206.21]
        else:
            bone_age = [3.02, 6.04, 9.05, 12.04, 18.22, 24.16, 30.96, 36.63, 43.50, 50.14, 60.06, 66.21, 78.50, 
                        89.30, 100.66, 113.86, 125.66, 137.86, 149.62, 162.28, 174.25, 183.62, 189.44]

        plt.plot(chronological_age, bone_age, color = 'r', linestyle = '--', marker='o', label = 'Brush Foundation Study')
        plt.plot(np.arange(0, max(chronological_age)+1), [pred_int] * (max(chronological_age)+1), color = 'b', linestyle = ':', label = 'prediction')
        plt.xlabel('Chronological Age [years]')
        plt.ylabel('Bone Age [months]')
        plt.legend()
        plt.title('Bone Age Growth Chart')
        plt.savefig('chart.png')
        growth_chart = np.array(Image.open('chart.png'))
        # growth_chart = np.random.random_integers(128, 256, image.shape)

        return Prediction(_create_dataset_for_image(item, 2, image_w_atlas),
                          _create_dataset_for_image(item, 3, heatmap),
                          _create_dataset_for_image(item, 4, growth_chart))


def _create_dataset_for_image(original: Dataset, series_num: int,
                              image: np.ndarray) -> Dataset:
    sop_uid = pydicom.uid.generate_uid(f'{PREDICTOR_UID_ROOT}2.{series_num}.')
    series_uid = pydicom.uid.generate_uid(f'{PREDICTOR_UID_ROOT}1.{series_num}.')

    file_metadata = FileMetaDataset()
    file_metadata.MediaStorageSOPClassUID = pydicom.uid.ComputedRadiographyImageStorage
    file_metadata.MediaStorageSOPInstanceUID = sop_uid
    file_metadata.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = copy.deepcopy(original)
    ds.Modality = 'OT'  # TODO: Check with THP what modality to use
    ds.SOPInstanceUID = sop_uid
    ds.SeriesInstanceUID = series_uid
    ds.ImageComments = ''
    ds.SeriesNumber = str(series_num)
    ds.InstanceNumber = '1'

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows = image.shape[0]
    ds.Columns = image.shape[1]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = image.tobytes()

    ds.file_meta = file_metadata
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    return ds
