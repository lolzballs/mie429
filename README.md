# MIE429 Capstone Project - Pediatric Bone Age Estimation

MIE429 is the 4th year capstone course for Engineering Science students majoring in Machine Intelligence at the University of Toronto in the class of 2T2. 

## Problem Statement and Overview
Our client, AI Development and Evaluation (AIDE) Lab at Trillium Health Partners (THP), has requested a machine learning (ML) solution to
improve the efficiency of radiologists while maintaining accuracy in bone age estimation. The
solution will utilize an input hand X-ray and may leverage the patient’s sex as well. Additionally,
this model must be deployed within an automated pipeline. The pipeline will fetch patient data in
the form of DICOM, the standard data format within medical imaging, from an image server
and return its predictions along with various visual tools to further help the radiologist. This project was supervised by clinical scientist and radiologist Dr. Benjamin Fine.

## Data
In order to train our model, we required a dataset with examples of X-ray images with their
associated bone age prediction from radiologists (ground truth), as well as the patients’ sex. To that end, our client suggested that we utilize the dataset provided by Radiological Society of North America (RSNA) in their 2017 Pediatric Bone Age Challenge.

Since there is no way to identify bone age objectively, multiple radiologists may predict different
bone ages for the same patient. To produce the singular prediction in the dataset, RSNA has
computed the ground truth values as a weighted consensus from 6 sources. As a reference,
the Mean Absolute Error (MAE) for these radiologists from the resulting ground truth was in the
range of 5 to 7 months. Because of this deviation in human accuracy and lack of alternative data
sources, we consider 5 months a lower bound in MAE for our model.

## Methodology and Results
In our solution, we explored the effects of a variety of data manipulation, transformation, and preprocessing techniques in the field of bone age estimation. Next, we trained CNNs to directly predict the bone age of the given input X-ray image. Our best model inspired from one of the top results from the 2017 RSNA Pediatric Bone Age Machine Learning Challenge achieved a validation MAE of 6.9 months, which exceeds the expectation of our clients and falls within the range of human radiologists. We also explored effects of modifying and filtering the dataset to assess how our solution would perform in production when image quality will differ from the ones used in training. 

In order to expose our model and associated outputs described previously in a useful manner to
radiologists, THP has requested that we implement our solution as a DICOM node. We have
created an open-source, fully-integrated application that receives inputs, performs inference with
our model, generates results/outputs, and provides the outputs

## Installation/Usage
This project was built using PyTorch with some data processing techniques using OpenCV.

If you do not need CUDA (i.e. you aren't doing training), we recommend using
the provided `requirements.txt`.

```
pip install requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Edit `app/__main__.py` to provide the location of the model, atlas, and your
DICOM addresses.

The app can be run by:
```
python app
```

## Development
We provide an `environment.yml` file to define python/package versions.

To set up a conda environment in `./env` using this yaml file:
```
conda env create --prefix env -f environment.yml
```

Then you can activate it with:
```
conda activate ./env
```

For more information about maintaining the `environment.yml` file please check
[the official conda reference](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).


