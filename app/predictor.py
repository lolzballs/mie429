import copy
import dataclasses
import queue
import threading
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydicom.dataset import Dataset, FileMetaDataset
import pydicom.uid
import torch
import torchvision


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

    def __init__(self, model_path: str, num_workers: int):
        self._model = torch.jit.load(model_path)
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
        # TODO: run through signature removal
        image = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
        age = self._model(image, torch.tensor(sex).unsqueeze(0))
        # TODO: XAI stuff
        heatmap = np.random.random_integers(128, 256, image.shape)
        return age, heatmap


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

        image = Image.fromarray(item.pixel_array, 'L')
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), f'{torch.round(prediction).int().item()}', fill="#fff",
                  font=self._predictor._font, stroke_width=5, stroke_fill="#000")
        image = np.array(image)
        growth_chart = np.random.random_integers(128, 256, image.shape)

        return Prediction(_create_dataset_for_image(item, 2, image),
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
