import os
import signal

import matplotlib
import pynetdicom

from predictor import Predictor
from dicom import SCP, SCU


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

    thread_count = os.cpu_count()
    predictor = Predictor('bilbily.pt',
                          'app/atlas',
                          'dataset_preprocessing/icon.png',
                          thread_count if thread_count is not None else 2)

    scu = SCU(predictor, ('15.222.138.226', 4242))
    scu.start()

    scp = SCP(predictor)
    scp.start(('0.0.0.0', 4242))

    # block until scu thread exists
    scu.join()
