import os
import sys
import itertools
import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from cnn_landslide_mapping.Utils import gdalCommonUtils as gutils
from cnn_landslide_mapping.Utils import IO as io
from cnn_landslide_mapping.Utils import tileHelpers as tileHelper

import tensorflow as tf
from tensorflow.keras import models
from osgeo import gdal, ogr, osr
import cv2


def open_and_rescale_to_baseResolution(img_path, shape, bbox, crop_to_bbox=None):
    img = gutils.readGDAL2numpy(img_path)
    img = tileHelper.rescaleInput(img, shape)
    if crop_to_bbox is not None:
        yMin, yMax, xMin, xMax = bbox
        img = img[yMin: yMax, xMin: xMax]
    return img


def normalize(image):
    new_img = np.array(image, dtype=np.float32)
    new_img /= 127.5
    new_img -= 1.
    return new_img


def normalizeArray(image):
    # post image
    image[:, :, :3] = normalize(image[:, :, :3])

    # pre image
    image[:, :, 3:6] = normalize(image[:, :, 3:6])

    # HS image
    image[:, :, 6:7] = normalize(image[:, :, 6:7])

    # Slope Image
    image[:, :, -1] = (image[:, :, -1] / 45.0) - 1.0

    return image


def scaleImg(img_arr, scale):

    scaled_arr = scale * (img_arr.astype(np.float64))

    max_value = np.max(scaled_arr, axis=2)

    saturate = max_value > 255

    scaled_arr[saturate, :] = 255

    return scaled_arr.astype(np.uint8)


def detect_landslides(model_path: str, output_path: str, raw_data_dict: dict, roi_path: str, debug: bool = False, brightness_scale: int = 25):

    if debug:
        print('TF Version -->', tf.__version__)
        print('GDAL Version -->', gdal.__version__)
        print('OPEN CV Version -->', cv2.__version__)
        print('CURRENT WORKING DIR -->', os.getcwd())

    # Prepare tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Prepare directories

    io.createDirectory(output_path, emptyExistingFiles=True, verbose=True)

    dem_path = raw_data_dict["dem_path"]
    hs_path = raw_data_dict["hs_path"]
    slope_path = raw_data_dict["slope_path"]

    post_image_path = raw_data_dict["post_image_path"]
    pre_image_path = raw_data_dict["pre_image_path"]
    no_data_mask = raw_data_dict["no_data_mask"]

    assert os.path.isfile(model_path)

    assert os.path.isfile(dem_path)
    assert os.path.isfile(hs_path)
    assert os.path.isfile(slope_path)

    if isinstance(post_image_path, str):
        assert os.path.isfile(post_image_path)
    else:
        assert os.path.isfile(post_image_path["B2"])
        assert os.path.isfile(post_image_path["B3"])
        assert os.path.isfile(post_image_path["B4"])

    if isinstance(pre_image_path, str):
        assert os.path.isfile(pre_image_path)
    else:
        assert os.path.isfile(pre_image_path["B2"])
        assert os.path.isfile(pre_image_path["B3"])
        assert os.path.isfile(pre_image_path["B4"])

    assert os.path.isfile(roi_path)

    # AOI dimensions

    bbox, binaryMask, newGeoT, proj = gutils.getBoundingBox(
        rasterPath=roi_path, returnBinaryMask=True)
    nscn, npix = binaryMask.shape

    # generate a mask from no data file
    if no_data_mask is None:
        maskImage = np.ones(shape=(nscn, npix))
    else:
        if os.path.isfile(no_data_mask):
            maskImage = 1 - open_and_rescale_to_baseResolution(
                img_path=no_data_mask, shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)
        else:
            maskImage = np.ones(shape=(nscn, npix))

    # append in mask file where test area Image == 0
    maskImage[binaryMask == 0] = 0

    maskFileName = os.path.join(*[output_path, 'MaskImage.tif'])
    gutils.writeNumpyArr2Geotiff(
        maskFileName, 1-maskImage, newGeoT, proj, GDAL_dtype=gdal.GDT_Byte, noDataValue=0)

    imageSize = 224
    overlapFactor = 2
    fetchSize = int(imageSize / 2)
    skipPx = int(imageSize / overlapFactor)

    # Generates all possible bounding boxes for tiling
    # The center location of every box is the anchor point defined in list 'locXY'
    y = [y for y in range(fetchSize + 1, nscn - fetchSize - 1, skipPx)]
    x = [x for x in range(fetchSize + 1, npix - fetchSize - 1, skipPx)]
    locXY = list(itertools.product(x, y))

    # extract all the valid boxes
    # i.e. which are in the study area and landslide mask
    # Mask Image == 1 for valid regions
    # threshold of 0.75 --> 75% region is valid
    validLocXY = [currLoc for currLoc in locXY if tileHelper.isValidTile(
        maskImage, imageSize, currLoc, threshold=0.50)]

    if isinstance(post_image_path, str):
        postImage = open_and_rescale_to_baseResolution(
            img_path=post_image_path, shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)
    else:
        postImage_sub = dict()
        postImage_sub["B2"] = open_and_rescale_to_baseResolution(
            img_path=post_image_path["B2"], shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)
        postImage_sub["B3"] = open_and_rescale_to_baseResolution(
            img_path=post_image_path["B3"], shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)
        postImage_sub["B4"] = open_and_rescale_to_baseResolution(
            img_path=post_image_path["B4"], shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)

        postImage = np.dstack(
            [postImage_sub["B4"], postImage_sub["B3"], postImage_sub["B2"]])

    if isinstance(pre_image_path, str):
        preImage = open_and_rescale_to_baseResolution(
            img_path=pre_image_path,  shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)
    else:
        preImage_sub = dict()
        preImage_sub["B2"] = open_and_rescale_to_baseResolution(
            img_path=pre_image_path["B2"], shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)
        preImage_sub["B3"] = open_and_rescale_to_baseResolution(
            img_path=pre_image_path["B3"], shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)
        preImage_sub["B4"] = open_and_rescale_to_baseResolution(
            img_path=pre_image_path["B4"], shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)

        preImage = np.dstack(
            [preImage_sub["B4"], preImage_sub["B3"], preImage_sub["B2"]])

    # scale the images

    postImage = scaleImg(postImage, brightness_scale)
    preImage = scaleImg(preImage, brightness_scale)

    hs = open_and_rescale_to_baseResolution(
        img_path=hs_path,    shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)
    slope = open_and_rescale_to_baseResolution(
        img_path=slope_path, shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)

    showMask = True

    # save images if in debug modes
    if debug:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 18), dpi=300)

        ax[0, 0].imshow(postImage)
        ax[0, 0].set_title('Sentinel-2 RGB (POST-EVENT)')

        ax[0, 1].imshow(preImage)
        ax[0, 1].set_title('Sentinel-2 RGB (PRE-EVENT)')

        ax[1, 0].imshow(hs, cmap='gray', vmin=0, vmax=255)
        ax[1, 0].set_title('HILLSHADE')

        ax[1, 1].imshow(slope, cmap='seismic', vmin=0, vmax=90)
        ax[1, 1].set_title('SLOPE')

        if showMask:
            maskForDisplay = 1 - maskImage.copy()
            maskForDisplay = np.float32(maskForDisplay)
            maskForDisplay[maskForDisplay == 0] = np.nan

            ax[0, 0].imshow(maskForDisplay, alpha=0.35,
                            cmap='cool', vmin=0, vmax=1)
            ax[0, 1].imshow(maskForDisplay, alpha=0.35,
                            cmap='cool', vmin=0, vmax=1)
            ax[1, 0].imshow(maskForDisplay, alpha=0.35,
                            cmap='cool', vmin=0, vmax=1)
            ax[1, 1].imshow(maskForDisplay, alpha=0.35,
                            cmap='cool', vmin=0, vmax=1)

        fig.savefig(os.path.join(output_path, "raw_data_plot.png"),
                    bbox_inches="tight")
        plt.close()

    # Prepare model input
    stack = []

    stack.append(postImage)
    stack.append(preImage)
    stack.append(hs[:, :, np.newaxis])
    stack.append(slope[:, :, np.newaxis])

    testImage = np.concatenate(stack, axis=2)

    testImage = np.nan_to_num(testImage)
    testImage = normalizeArray(testImage)

    assert testImage.max() <= 1
    assert testImage.min() >= -1

    inputChannel = testImage.shape[2]

    predictMask = np.zeros((testImage.shape[0], testImage.shape[1]), np.uint8)

    fetchSize_half = int(fetchSize/2)

    # Load tf model

    model = models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Forward pass for each sub-window of the figure

    for x, y in tqdm(validLocXY):

        img = testImage[y - fetchSize: y + fetchSize,
                        x - fetchSize: x + fetchSize, :]

        img = img.reshape(1, imageSize, imageSize, inputChannel)
        predict_image_list = [img[:, :, :, 3:6],  # pre-image
                              img[:, :, :, :3],  # post-image
                              img[:, :, :, -2:]  # topo-image
                              ]
        predicted_label = model.predict(predict_image_list)[0][0]

        predictMask[y - fetchSize_half: y + fetchSize_half, x - fetchSize_half: x +
                    fetchSize_half] = predicted_label[fetchSize_half: -fetchSize_half, fetchSize_half: -fetchSize_half, 0] * 100

    # Model output post-processing and save

    predictMask = predictMask * maskImage
    gutils.writeNumpyArr2Geotiff(output_path + '/Predict_LS_Conf.tif',
                                 predictMask, newGeoT, proj, GDAL_dtype=gdal.GDT_Byte, noDataValue=0)

    threshold = 50
    predictMask_highConf = predictMask.copy()
    predictMask_highConf[predictMask_highConf < threshold] = 0
    predictMask_highConf[predictMask_highConf >= threshold] = 1

    rasterPath = output_path + \
        '/Predict_LS_HighConf_{}.tif'.format(str(threshold))
    gutils.writeNumpyArr2Geotiff(
        rasterPath, predictMask_highConf, newGeoT, proj, GDAL_dtype=gdal.GDT_Byte, noDataValue=0)

    showMask = True

    LSForDisplay = predictMask_highConf.copy()
    LSForDisplay = np.float32(LSForDisplay)
    LSForDisplay[LSForDisplay == 0] = np.nan

    # Save predicted regions as images if in debug mode
    if debug:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15), dpi=300)

        ax[0].imshow(postImage)
        ax[0].set_title('Sentinel-2 RGB (POST-EVENT)')
        ax[0].axis('off')

        ax[1].imshow(postImage, alpha=0.35)
        ax[1].imshow(LSForDisplay,
                     alpha=1, cmap='bwr', vmin=0, vmax=1)
        ax[1].set_title('MAPPED LANDSLIDE IN RED')
        ax[1].axis('off')

        ax[2].imshow(preImage)
        ax[2].set_title('Sentinel-2 RGB (PRE-EVENT)')
        ax[2].axis('off')

        if showMask:
            maskForDisplay = 1 - maskImage.copy()
            maskForDisplay = np.float32(maskForDisplay)
            maskForDisplay[maskForDisplay == 0] = np.nan

            ax[0].imshow(maskForDisplay,
                         alpha=0.35, cmap='cool', vmin=0, vmax=1)
            ax[1].imshow(maskForDisplay,
                         alpha=0.35, cmap='cool', vmin=0, vmax=1)
            ax[2].imshow(maskForDisplay,
                         alpha=0.35, cmap='cool', vmin=0, vmax=1)

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "prediction_plot.png"),
                bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    raw_data_dict = dict()

    raw_data_dict["dem_path"] = "Japan_data/dem/download.DSM.tif"
    raw_data_dict["hs_path"] = "Japan_data/hs/download.hillshade.tif"
    raw_data_dict["slope_path"] = "Japan_data/slope/download.slope.tif"

    raw_data_dict["post_image_path"] = dict()
    raw_data_dict["post_image_path"]["B2"] = "Japan_data/post-event/download.B2_uint8.tif"
    raw_data_dict["post_image_path"]["B3"] = "Japan_data/post-event/download.B3_uint8.tif"
    raw_data_dict["post_image_path"]["B4"] = "Japan_data/post-event/download.B4_uint8.tif"

    raw_data_dict["pre_image_path"] = dict()
    raw_data_dict["pre_image_path"]["B2"] = "Japan_data/pre-event/download.B2_uint8.tif"
    raw_data_dict["pre_image_path"]["B3"] = "Japan_data/pre-event/download.B3_uint8.tif"
    raw_data_dict["pre_image_path"]["B4"] = "Japan_data/pre-event/download.B4_uint8.tif"

    raw_data_dict["no_data_mask"] = None

    detect_landslides(model_path="M_ALL_006.hdf5", output_path="Mapping_results_jp",
                      raw_data_dict=raw_data_dict, roi_path="Japan_data/aoi_S2.tif", debug=True)
