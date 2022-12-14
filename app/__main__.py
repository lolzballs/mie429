import os
import signal

import matplotlib
import pynetdicom

from predictor import Predictor
from dicom import SCP, SCU


MODEL_LOCATION             = 'resources/bilbily.pt'
ATLAS_LOCATION             = 'resources/atlas'
FONT_LOCATION              = 'resources/Arial.ttf'
EXAMPLE_SIGNATURE_LOCATION = 'dataset_preprocessing/icon.png'
SCP_BIND_ADDRESS           = ('0.0.0.0', 4242)
REMOTE_DICOM_SERVER        = ('15.222.138.226', 4242)


def sigint_handler(signum, stack_frame):
    global stopping
    if not stopping:
        stopping = True
        print('stopping scp')
        scp.stop()
        print('stopping predictor')
        predictor.stop()
        print('stopping scu')
        scu.stop()


if __name__ == '__main__':
    if os.environ.get('debug') is not None:
        pynetdicom.debug_logger()

    matplotlib.use('agg')

    stopping = False
    signal.signal(signal.SIGINT, sigint_handler)

    # XAI requires model to be stored per-thread, so it'll cost too much RAM
    thread_count = 1
    predictor = Predictor(MODEL_LOCATION,
                          ATLAS_LOCATION,
                          EXAMPLE_SIGNATURE_LOCATION,
                          FONT_LOCATION,
                          thread_count if thread_count is not None else 2)

    scu = SCU(predictor, REMOTE_DICOM_SERVER)
    scu.start()

    scp = SCP(predictor)
    scp.start(SCP_BIND_ADDRESS)

    # block until scu thread exists
    scu.join()
