from cnn_mapping.predictor import detect_landslides

if __name__ == "__main__":
    raw_data_dict = dict()

    raw_data_dict["dem_path"] = "RawData/AOI_E_S2-S2/DEM/DEM_030.tif"
    raw_data_dict["hs_path"] = "RawData/AOI_E_S2-S2/DEM/DEM_030_HILLSHADE.tif"
    raw_data_dict["slope_path"] = "RawData/AOI_E_S2-S2/DEM/DEM_030_SLOPE.tif"

    raw_data_dict["post_image_path"] = "RawData/AOI_E_S2-S2/fromOptical/JIUZ_POST_S2_RGB_010_UINT8.tif"
    raw_data_dict["pre_image_path"] = "RawData/AOI_E_S2-S2/fromOptical/JIUZ_PRE_S2_RGB_010_UINT8.tif"
    raw_data_dict["no_data_mask"] = "RawData/AOI_E_S2-S2/noDataMask/SNOW_CLOUD_MASK_010.tif"

    detect_landslides(model_path="M_ALL_006.hdf5", output_path="Mapping_results",
                      raw_data_dict=raw_data_dict, roi_path="RawData/AOI_E_S2-S2/testBoundary/Test_006.tif", debug=True)
