# MIE429 Capstone Project

## Installation/Usage
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
