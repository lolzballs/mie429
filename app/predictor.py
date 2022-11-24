import copy
import dataclasses
import queue
import os
import threading
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
import pydicom.uid
from scipy import interpolate
import torch
import torchvision

from atlas import Atlas
from growth_chart import GrowthChart, GrowthChartInput

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..',
                             'dataset_preprocessing'))
from image_matcher import ImageMatcher


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


class Predictor:
    _process_queue: queue.Queue[Dataset]
    _result_queue: queue.Queue[Prediction]
    _model: torch.nn.Module

    def __init__(self, model_path: str, atlas_path: str, icon_path: str,
                 num_workers: int):
        self._atlas = Atlas(atlas_path)
        self._image_matcher = ImageMatcher(cv2.imread(icon_path))
        self._growth_chart = GrowthChart()

        self._model = torch.jit.load(model_path)
        self._model.eval()
        self._model_final_layer = self._model._modules['model']\
            ._modules['inception']._modules['Mixed_7c']

        self._running = True
        self._process_queue = queue.Queue()
        self._result_queue = queue.Queue()
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
        self._growth_chart.stop()

    def _run_model(self, image: np.ndarray, sex: int) \
            -> Tuple[torch.Tensor, np.ndarray]:
        # remove signatures
        self._image_matcher.upload_search_image(image)
        _, result_img = self._image_matcher.find_object(padding_X=125,
                                                        padding_Y=175,
                                                        draw_box=False)
        original_image = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
        if result_img is not None:
            image = result_img

        image = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
        age = self._model(image, torch.tensor(sex).unsqueeze(0).float())
        heatmap = self._get_heatmap(original_image, self._model_final_layer.activations)
        return age, heatmap

    def _get_heatmap(self, image, activated_features) -> np.ndarray:
        weight_softmax_params = list(self._model._modules['model']._modules['fc'].parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        final_fc_weights = weight_softmax_params[2].data.numpy()

        image = image.expand(-1,3,-1,-1).squeeze().numpy().transpose((1, 2, 0))
        image = image * 255
        heatmap_only = self._get_CAM(activated_features, weight_softmax, final_fc_weights)
        heatmap = cv2.resize(heatmap_only, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + image * 0.6
        superimposed_img = superimposed_img.astype(np.uint8)
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
        image_w_atlas = self._predictor._atlas.render(item.pixel_array,
                                                      item.PatientSex,
                                                      pred_int)

        chart = self._predictor._growth_chart.plot_chart(GrowthChartInput(item.PatientSex, pred_int))

        return Prediction(_create_dataset_for_image(item, 2, image_w_atlas),
                          _create_dataset_for_image(item, 3, heatmap),
                          _create_dataset_for_image(item, 4, chart))


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
    ds.PhotometricInterpretation = 'RGB' if series_num > 2 else 'MONOCHROME2'
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
