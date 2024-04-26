#load libraries
import os
import shutil
import numpy as np
import tensorflow as tf
import cv2
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import pkg_resources
pkg_resources.require("suntime==1.2.5") #important: suntime v1.2.5 required
from suntime import Sun
from datetime import datetime

#constants
TENSORFLOW_MODEL = 'data/TensorFlow_model.ckpt'
NETCDF_FILE      = 'data/ceilonet_chm15k_backsct_la1_t12s_v1.0_06348_A20210519.nc'
IMG_MODEL_INPUT  = 'data/latest_model_input.png'
FIG_MODEL_OUTPUT = 'data/results_{}.png'
CEILOMETER_LAT   = 51.965608
CEILOMETER_LON   = 4.896306
PIXELS_TEMPORAL  = 224
PIXELS_SPATIAL   = 224
SPATIAL_RES      = 10
MAX_RCS          = 1e6
OUTPUT_CLASSES   = 1

#functions
def upsample(filters, size):
  """Upsamples an input via Conv2DTranspose => Batchnorm => Relu"""
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                             padding='same',
                                             kernel_initializer=initializer,
                                             use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.ReLU())
  
  return result

def create_model_architecture(output_channels):
  """"This function creates the Deep-Pathfinder model architecture, visualised
  in Fig. 4 of the corresponding AMT paper."""
  #start point
  input_rcs = tf.keras.layers.Input(shape=(224, 224, 3), name='rcs_image')
  input_nighttime = tf.keras.layers.Input(shape=(1,), name='nighttime_indicator')

  #create lightweight encoder architecture based on MobileNet
  base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3],
                                                 include_top=False, weights=None)

  #use the activations of these layers for horizontal skip connections
  layer_names = [
      'block_1_expand_relu',   # 112x112
      'block_3_expand_relu',   # 56x56
      'block_6_expand_relu',   # 28x28
      'block_13_expand_relu',  # 14x14
      'block_16_project',      # 7x7
  ]
  base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

  #create the feature extraction model
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

  #create decoder
  up_stack = [
      upsample(512, 3),  # 7x7   -> 14x14
      upsample(256, 3),  # 14x14 -> 28x28
      upsample(128, 3),  # 28x28 -> 56x56
      upsample(64, 3),   # 56x56 -> 112x112
  ]

  #connect encoder and decoder
  skips = down_stack(input_rcs, training=False)
  x = skips[-1]
  skips = reversed(skips[:-1])

  #merge in the nighttime variable as an additional feature
  nighttime_layer = tf.repeat(input_nighttime, repeats=49, axis=1)
  nighttime_layer = tf.reshape(nighttime_layer, shape=(-1, 7, 7, 1))
  concat = tf.keras.layers.Concatenate()
  x = concat([x, nighttime_layer])

  #increasing dimensions and establishing skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  #final model layer to construct output image
  last_layer = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same', name='mask_image')  #112x112 -> 224x224

  x = last_layer(x)

  return tf.keras.Model(inputs=[input_rcs, input_nighttime], outputs=x)

def get_blh(altitude_predictions):
  """This function extracts the most likely boundary layer height from the
  predicted mask, at a single time point. See AMT paper for details."""
  loss = [None] * len(altitude_predictions)

  error_a = 1 - altitude_predictions
  error_b = np.flip(altitude_predictions)
  error_a_cumulative = np.cumsum(error_a)
  error_b_cumulative = np.cumsum(error_b)
  error_b_cumulative = np.flip(error_b_cumulative)
  loss = error_a_cumulative + error_b_cumulative - error_a

  index_blh = np.argmin(loss)

  return (PIXELS_SPATIAL - index_blh) * SPATIAL_RES

def main():
  #create architecture and load saved weights
  model = create_model_architecture(output_channels=OUTPUT_CLASSES)
  model.load_weights(TENSORFLOW_MODEL)

  #load ceilometer data from NetCDF file
  input_data = xr.load_dataset(NETCDF_FILE)
  beta_raw = input_data.beta_raw.T.values

  #pre-process rcs values to [0,1] range
  rcs = np.maximum(beta_raw, 0)
  rcs = np.minimum(rcs, MAX_RCS)
  rcs = rcs / MAX_RCS

  #cap altitude to lower atmosphere and select most recent time period
  total_time = len(rcs[0,:])
  if total_time < PIXELS_TEMPORAL:
      print("Insufficient time steps in input data to apply algorithm.")
  rcs = rcs[:PIXELS_SPATIAL, (total_time - PIXELS_TEMPORAL):total_time]
  rcs = np.flip(rcs, axis=0)

  #reverse colour scale and store as 16-bit grayscale image (for testing purposes)
  rcs = cv2.normalize(-rcs, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX,
                      dtype=cv2.CV_32F)
  rcs = rcs.astype(np.uint16)
  rcs = cv2.cvtColor(rcs, cv2.COLOR_GRAY2BGR)
  cv2.imwrite(IMG_MODEL_INPUT, rcs)

  #load image
  rcs_img = tf.io.read_file(IMG_MODEL_INPUT)
  rcs_img = tf.io.decode_png(rcs_img, channels=1, dtype=tf.dtypes.uint16)
  rcs_img = tf.cast(rcs_img, tf.float32) / 65535.0
  rcs_img = tf.image.grayscale_to_rgb(rcs_img)

  #get current sunrise and sunset times in UTC
  dt = datetime.now(tz=pytz.timezone('Europe/Amsterdam'))
  sun = Sun(CEILOMETER_LAT, CEILOMETER_LON)
  sunrise = sun.get_sunrise_time(dt)
  sunset  = sun.get_sunset_time(dt)

  #note that this script is meant for real-time application using your latest
  #NetCDF data, so the current time is used to set the nighttime indicator
  if dt > sunrise and dt < sunset:
    nighttime_indicator = 0
  else:
    nighttime_indicator = 1

  #combine RCS image and nighttime indicator
  input_rcs = tf.expand_dims(rcs_img, axis=0)
  input_NI  = tf.expand_dims(nighttime_indicator, axis=0)
  model_input = (input_rcs, input_NI)

  #model inference using calibrated model
  mask = model.predict(model_input)
  mask = tf.math.sigmoid(mask)
  mask = mask[0,:,:,0]

  #extract boundary layer height (BLH) for each time step independently
  num_cols = mask.shape[1]
  blh = [None] * num_cols
  for i in range(num_cols):
    blh[i] = get_blh(mask[:,i])

  #pad start with NA, as predictions were only made for the most recent data
  padding = np.repeat(np.nan, (total_time - PIXELS_TEMPORAL))
  blh_padded = np.concatenate((padding, blh))

  #copy original netCDF file
  netCDF_output_path = os.path.splitext(NETCDF_FILE)[0] + '_Deep-Pathfinder.nc'
  shutil.copyfile(NETCDF_FILE, netCDF_output_path)

  #add DeepPathfinder predictions to netCDF file
  with nc.Dataset(netCDF_output_path, 'a') as ds:
    deepPathfinder = ds.createVariable('DeepPathfinder', 'f4', ('time',))
    deepPathfinder.units = 'm'
    deepPathfinder.long_name = 'DeepPathfinder_BLH_prediction'
    deepPathfinder[:] = blh_padded

  #create combined plot of DeepPathfinder and manufacturer BLH estimates
  all_data = xr.load_dataset(netCDF_output_path)
  filtered_data = all_data.where(~np.isnan(all_data.DeepPathfinder), drop=True)

  #extract data for plotting
  netcdf_time        = filtered_data.time.values
  netcdf_altitude    = filtered_data.range.values
  netcdf_rcs         = filtered_data.beta_raw.T.values
  blh_DeepPathfinder = filtered_data.DeepPathfinder
  blh_manufacturer   = filtered_data.pbl.values
  blh_manufacturer[blh_manufacturer <= 0] = np.nan

  #create plot
  saved_font_size = plt.rcParams['font.size']
  plt.rcParams.update({'font.size': 20})
  myFmt = mdates.DateFormatter('%H:%M')
  plt.figure(figsize=(16, 12))
  #fig = plt.figure(figsize=(16, 12))
  plt.pcolormesh(netcdf_time, netcdf_altitude, netcdf_rcs, cmap='terrain',
                 shading='nearest')
  plt.plot(netcdf_time, blh_manufacturer[:,0], '.r-')
  plt.plot(netcdf_time, blh_manufacturer[:,1], '.y')
  plt.plot(netcdf_time, blh_manufacturer[:,2], '.m')
  plt.plot(netcdf_time, blh_DeepPathfinder,'.k-')
  plt.clim([0,1e6])
  plt.ylim([0,4000])
  plt.colorbar(label='Range-corrected signal_1064nm [a.u.]')
  plt.xlabel(filtered_data.time.attrs['long_name'])
  plt.ylabel(filtered_data.range.attrs['long_name'] + ' [' +
             filtered_data.range.attrs['units'] + ']')
  plt.gca().xaxis.set_major_formatter(myFmt)
  plt.gca().xaxis.set_major_locator(mdates.HourLocator())
  plt.savefig(FIG_MODEL_OUTPUT.format(dt.strftime("%d-%m-%Y_%H%M%S")),
              bbox_inches='tight')
  plt.close()
  plt.rcParams.update({'font.size': saved_font_size})

if __name__ == "__main__":
  main()