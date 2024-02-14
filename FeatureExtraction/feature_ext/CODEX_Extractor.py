"""

Main object for CODEX nuclei segmentation and feature extraction


"""

import os
import sys

import numpy as np

import cv2

from skimage.morphology import remove_small_objects, clear_border, remove_small_holes
from skimage.segmentation import watershed, expand_labels
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from skimage.draw import polygon
import json
from tqdm import tqdm

import large_image

class CODEXtractor:
    def __init__(self,
                 image_id,
                 region,
                 seg_params,
                 gc):
        
        self.gc = gc
        self.image_id = image_id
        self.region = region
        self.seg_params = seg_params

        # Getting image information
        self.image_info = gc.get(f'/item/{self.image_id}')

        self.tile_source = large_image.getTileSource(f'/{self.image_info["name"]}')

    def get_image_region(self,frame,region_coords):

        # Region coords in this case should be in the form [minx, miny, maxx, maxy] or left, top, right, bottom
        image_region = self.tile_source.getRegion(
            region = {
                'left': region_coords[0],
                'top': region_coords[1],
                'right': region_coords[2],
                'bottom': region_coords[3]
            },
            frame = frame,
            format = large_image.constants.TILE_FORMAT_NUMPY
        )

        return image_region

    def get_nuclei(self, image_region):

        # For an input image region, segment the nuclei.
        # Assumes grayscale input image with nuclei being lighter than background
        # returns labeled mask

        thresh_image = image_region.copy()

        # Thresholding grayscale image
        thresh_image[thresh_image<=self.seg_params['threshold']] = 0
        thresh_image[thresh_image>0] = 1

        # Post-processing thresholded binary image
        remove_holes = remove_small_holes(thresh_image>0,area_threshold = 10)
        remove_holes = remove_holes>0
        # Watershed transform for splitting overlapping nuclei
        distance_transform = ndi.distance_transform_edt(remove_holes)
        labeled_mask, _ = ndi.label(remove_holes)
        coords = peak_local_max(distance_transform,footprint=np.ones((3,3)),labels = labeled_mask)
        watershed_mask = np.zeros(distance_transform.shape,dtype=bool)
        watershed_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(watershed_mask)

        watershedded = watershed(-distance_transform,markers,mask=remove_holes)
        watershedded = watershedded

        # Removing any small objects
        processed_nuclei = remove_small_objects(watershedded,self.seg_params['min_size'])

        # Removing nuclei that touch the borders of the image region
        processed_nuclei = clear_border(processed_nuclei)

        processed_nuclei = ndi.label(processed_nuclei)

        # Getting mask of cell "cytoplasm" associated with each nucleus (can subtract processed_nuclei to get only the labeled "cytoplasm")
        cytoplasm = expand_labels(processed_nuclei, distance = self.seg_params['cyto_pixels'])

        return processed_nuclei, cytoplasm















