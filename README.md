# Deep-Pathfinder
This repository provides an implementation of the Deep-Pathfinder algorithm, extracting mixing boundary layer height information from Lufft CHM15k ceilometer data using computer vision techniques.

Note that this script is designed for real-time application, so please replace the NetCDF file with your latest observation data. For real-time inference, the value of the nighttime indicator is set based on the current time when the script is run. If you are interested in analysing historical data, this should be changed.

For full details of the methodology and citation, please refer to: Wijnands, J.S., Apituley, A., Alves Gouveia, D., Noteboom, J.W. (2024). Deep-Pathfinder: a boundary layer height detection algorithm based on image segmentation. Atmospheric Measurement Techniques, doi: https://doi.org/10.5194/amt-2023-80

# Installation instructions

This script has several dependencies. The following example shows how to set up a suitable Anaconda environment:

```
conda create --name Deep-Pathfinder python=3.10 -y
conda activate Deep-Pathfinder
pip install "tensorflow==2.10"
pip install opencv-python
conda install -y netCDF4
conda install -y xarray
conda install -y matplotlib
pip install "suntime==1.2.5"

```

To apply the Deep-Pathfinder algorithm using the last 45 minutes of ceilometer data in the supplied NetCDF file, run the command below. Results will be written to the 'data' folder.

```
python Deep-Pathfinder_inference.py
```
