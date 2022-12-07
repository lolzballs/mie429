import queue
import threading
from typing import Tuple

import pydicom.uid
import pynetdicom
from pynetdicom.sop_class import ComputedRadiographyImageStorage

from predictor import Predictor


class SCU(threading.Thread):
    def __init__(self, predictor: Predictor, remote: Tuple[str, int]):
        threading.Thread.__init__(self, name='SCU_Thread')

        self._predictor = predictor
        self._remote = remote
        self._running = False

        self._ae = pynetdicom.AE()
        self._ae.add_requested_context(ComputedRadiographyImageStorage,
                                       pydicom.uid.ExplicitVRLittleEndian)

    def start(self):
        self._running = True

        threading.Thread.start(self)

    def stop(self):
        self._running = False
        threading.Thread.join(self)

    def run(self):
        while self._running:
            try:
                prediction = self._predictor.get_result(1)
            except queue.Empty:
                continue

            assoc = self._ae.associate(self._remote[0], self._remote[1])
            if not assoc.is_established:
                print('could not associate with remote, data may be lost')
                continue

            status = assoc.send_c_store(prediction.prediction)
            print('C-STORE (prediction) status: 0x{0:04X}'
                  .format(status.Status))
            status = assoc.send_c_store(prediction.heatmap)
            print('C-STORE (heatmap) status: 0x{0:04X}'
                  .format(status.Status))
            status = assoc.send_c_store(prediction.growth_chart)
            print('C-STORE (growth_chart) status: 0x{0:04X}'
                  .format(status.Status))
            assoc.release()


class SCP:
    SUPPORTED_TS = [
        pydicom.uid.ExplicitVRLittleEndian,
        pydicom.uid.ExplicitVRBigEndian,
        *pydicom.uid.JPEG2000TransferSyntaxes
    ]

    def __init__(self, predictor: Predictor):
        self._predictor = predictor

        self._ae = pynetdicom.AE()
        for ts in self.SUPPORTED_TS:
            self._ae.add_supported_context(ComputedRadiographyImageStorage, ts)

        self._event_handlers = [
            (pynetdicom.evt.EVT_C_STORE, self._handle_store),
        ]

    def start(self, address: Tuple[str, int]):
        self._ae.start_server(address, block=False,
                              evt_handlers=self._event_handlers)

    def stop(self):
        self._ae.shutdown()

    def _handle_store(self, event: pynetdicom.evt.Event):
        ds = event.dataset
        ds.file_meta = event.file_meta

        if ds.StudyDescription != 'XR BONE AGE LEFT 1 VIEW':
            print(f'ignoring study {ds.StudyInstanceUID} since it does not '
                  'have study description "XR BONE AGE LEFT 1 VIEW"')
        else:
            self._predictor.predict(ds)

        return 0x0000
