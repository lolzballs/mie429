import os

import matplotlib
import pydicom.uid
from pydicom.dataset import Dataset
import pynetdicom
from pynetdicom.sop_class import (
    ComputedRadiographyImageStorage,
    StudyRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelGet,
)

from predictor import Predictor


def handle_store(event: pynetdicom.evt.Event):
    ds = event.dataset
    ds.file_meta = event.file_meta
    predictor.predict(ds)
    return 0x0000


if __name__ == '__main__':
    if os.environ.get('debug') is not None:
        pynetdicom.debug_logger()

    matplotlib.use('agg')

    study_instance_uids = []

    thread_count = os.cpu_count()
    predictor = Predictor('bilbily.pt',
                          'app/atlas',
                          'dataset_preprocessing/icon.png',
                          thread_count if thread_count is not None else 2)
    ae = pynetdicom.AE()

    # query for relevant bone age studies
    ds = Dataset()
    ds.QueryRetrieveLevel = 'STUDY'
    ds.PatientName = ''
    ds.AccessionNumber = ''
    ds.StudyInstanceUID = ''
    ds.StudyDate = ''
    ds.StudyTime = ''
    ds.StudyDescription = 'XR BONE AGE LEFT 1 VIEW'
    ds.NumberOfStudyRelatedSeries = None
    ds.PatientName = ''

    # Associate with the peer AE
    event_handlers = [
        (pynetdicom.evt.EVT_C_STORE, handle_store),
    ]
    ae.add_requested_context(ComputedRadiographyImageStorage,
                             pydicom.uid.ExplicitVRLittleEndian)
    for ts in pydicom.uid.JPEG2000TransferSyntaxes:
        ae.add_requested_context(ComputedRadiographyImageStorage, ts)
    ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)
    ae.add_requested_context(StudyRootQueryRetrieveInformationModelGet)
    role = pynetdicom.build_role(ComputedRadiographyImageStorage,
                                 scp_role=True, scu_role=True)
    assoc = ae.associate('15.222.138.226', 4242,
                         evt_handlers=event_handlers,
                         ext_neg=[role])

    # Send the C-FIND request to get relevant results (like SELECT)
    responses = assoc.send_c_find(ds, StudyRootQueryRetrieveInformationModelFind)
    for (status, identifier) in responses:
        if status:
            print('C-FIND query status: 0x{0:04X}'.format(status.Status))
            if status.Status == 0xFF00:  # Pending
                if identifier.NumberOfStudyRelatedSeries == '1':
                    study_instance_uids.append(identifier.StudyInstanceUID)
                else:
                    print(f'skipping {identifier.StudyInstanceUID} as it has '
                          'more than one series')
        else:
            print('Connection timed out, was aborted or received invalid response')

    print(f'found {len(study_instance_uids)} studies')
    # This gets us the study instance UIDs we can use to run a C-GET to fetch the DICOM files
    for study_instance_uid in study_instance_uids:
        ds = Dataset()
        ds.Modality = 'CR'
        ds.QueryRetrieveLevel = 'STUDY'
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = None
        responses = assoc.send_c_get(ds, StudyRootQueryRetrieveInformationModelGet)
        for (status, identifier) in responses:
            print('C-GET query status: 0x{0:04X}'.format(status.Status))

    for study_instance_uid in study_instance_uids:
        prediction = predictor.get_result()
        print(prediction.prediction.SeriesInstanceUID)
        responses = assoc.send_c_store(prediction.prediction)
        print(prediction.heatmap.SeriesInstanceUID)
        responses = assoc.send_c_store(prediction.heatmap)
        print(prediction.growth_chart.SeriesInstanceUID)
        responses = assoc.send_c_store(prediction.growth_chart)
        print('C-STORE', responses.Status)

    predictor.stop()
    assoc.release()
