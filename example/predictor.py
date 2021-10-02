from cnn_landslide_mapping.predictor import detect_landslides



def example_orig():
    raw_data_dict = dict()

    raw_data_dict["dem_path"] = "RawData/AOI_E_S2-S2/DEM/DEM_030.tif"
    raw_data_dict["hs_path"] = "RawData/AOI_E_S2-S2/DEM/DEM_030_HILLSHADE.tif"
    raw_data_dict["slope_path"] = "RawData/AOI_E_S2-S2/DEM/DEM_030_SLOPE.tif"

    raw_data_dict["post_image_path"] = "RawData/AOI_E_S2-S2/fromOptical/JIUZ_POST_S2_RGB_010_UINT8.tif"
    raw_data_dict["pre_image_path"] = "RawData/AOI_E_S2-S2/fromOptical/JIUZ_PRE_S2_RGB_010_UINT8.tif"
    raw_data_dict["no_data_mask"] = "RawData/AOI_E_S2-S2/noDataMask/SNOW_CLOUD_MASK_010.tif"

    detect_landslides(model_path="M_ALL_006.hdf5", output_path="Mapping_results",
                      raw_data_dict=raw_data_dict, roi_path="RawData/AOI_E_S2-S2/testBoundary/Test_006.tif", debug=True)

def example_jp():
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
if __name__ == "__main__":
    example_orig()
    example_jp()
