"""

Main object for CODEX nuclei segmentation and feature extraction


"""

import os
import sys

import numpy as np

from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import watershed, expand_labels, clear_border
from skimage.draw import polygon
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

import pandas as pd

from typing import List, Union

import large_image
from PIL import Image
import requests
from io import BytesIO

import wsi_annotations_kit.wsi_annotations_kit as wak

from tqdm import tqdm

class CODEXtractor:
    def __init__(self,
                 image_id,
                 seg_params,
                 gc):
        
        self.gc = gc

        self.user_token = self.gc.get('/token/session')['token']
        self.image_id = image_id
        self.seg_params = seg_params

        # Getting image information
        self.image_info = gc.get(f'/item/{self.image_id}')

        self.tile_source = large_image.getTileSource(f'/{self.image_info["name"]}')
        self.tile_metadata = self.tile_source.getMetadata()

    def get_image_region(self,coords_list,frame_index):

        # Region coords in this case should be in the form [minx, miny, maxx, maxy] or left, top, right, bottom
        """
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
        """
        image_region = np.array(Image.open(BytesIO(requests.get(self.gc.urlBase+f'/item/{self.image_id}/tiles/region?token={self.user_token}&frame={frame_index}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}').content)))

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
        processed_nuclei, _ = ndi.label(processed_nuclei)

        # Getting mask of cell "cytoplasm" associated with each nucleus (can subtract processed_nuclei to get only the labeled "cytoplasm")
        cytoplasm = expand_labels(processed_nuclei, distance = self.seg_params['cyto_pixels'])

        return processed_nuclei, cytoplasm

    def make_annotations(self, labeled_mask, region_coords):

        # Making large-image formatted annotations from an input label mask
        binary_mask = labeled_mask.copy()
        binary_mask[binary_mask>0] = 1

        # Adding the annotation mask to a wak annotation object
        annotations = wak.Annotation()
        annotations.add_mask(
            mask = binary_mask,
            box_crs = [region_coords[0],region_coords[1]],
            mask_type = 'labeled'
        )

        # Conversion to JSON
        json_annotations = wak.Histomics(annotations).json

        return json_annotations

    def get_intensity_features(self, region_coords:list, return_type: Union[List [str], str]):

        # Main feature extraction function for pulling a region from the image, segmenting nuclei, calculating features, and returning some formatted object
        # return_type can be a string or list of strings 

        # Step 1: Segmenting the nuclei
        """
        nuclei_region, _ = self.tile_source.getRegion(
            frame = self.seg_params['frame'],
            region = {
                'left': region_coords[0],
                'top': region_coords[1],
                'right': region_coords[2],
                'bottom': region_coords[3]
            },
            output = dict(maxWidth = 2000),
            format = large_image.constants.TILE_FORMAT_NUMPY
        )
        """
        nuclei_region = self.get_image_region(region_coords,self.seg_params['frame'])

        nuclei_mask, cytoplasm_mask = self.get_nuclei(nuclei_region)

        # If there are no nuclei, don't do the rest here
        if np.sum(nuclei_mask)==0:
            print('No nuclei found :(')
            return None, None

        # Step 2: Creating annotations
        annotations_json = self.make_annotations(cytoplasm_mask, region_coords)

        # Step 3: Get list of image regions from each frame
        frame_list = []
        for f in range(0,len(self.tile_metadata['frames'])-1):
            
            """
            frame_region = self.tile_source.getRegion(
                frame = f,
                region = {
                    'left': region_coords[0],
                    'top': region_coords[1],
                    'right': region_coords[2],
                    'bottom': region_coords[3]
                },
                format = large_image.constants.TILE_FORMAT_NUMPY
            )
            """
            frame_region = self.get_image_region(region_coords,f)
            frame_list.append(frame_region)
        
        # Creating an array that should be regionY x regionX x nFrames
        frame_array = np.array(frame_list)

        # Step 4: Iterating through nuclei and getting frame statistics
        feature_list = [
            'Mean_Channels',
            'Std_Channels',
            'Max_Channels',
            'Min_Channels',
            'Median_Channels'
        ]

        print(f'Found: {len(annotations_json[0]["annotation"]["elements"])} Nuclei')
        if len(annotations_json[0]['annotation']['elements'])>0:
            for nuc_idx,nuc in tqdm(enumerate(annotations_json[0]['annotation']['elements'])):
                
                # Getting a binary mask of this specific nucleus and its cytoplasm
                #specific_nuc_mask = cytoplasm_mask.copy()
                #specific_nuc_mask[specific_nuc_mask != nuc] = 0
                #specific_nuc_mask[specific_nuc_mask>0] = 1

                # Instead of creating the mask from the cytoplasm mask, creating from the annotation
                # This insures alignment in the event that a nucleus is not a valid shape
                specific_nuc_mask = np.zeros_like(nuclei_mask)
                adjusted_coords = [[i[0]-region_coords[0],i[1]-region_coords[1]] for i in nuc['points']]
                row_coords = [i[1] for i in adjusted_coords]
                col_coords = [i[0] for i in adjusted_coords]
                row, col = polygon(row_coords,col_coords)
                specific_nuc_mask[row,col] = 1

                # Masking the frame_array
                masked_frames = np.where(specific_nuc_mask>0,frame_array.copy(),0)

                # Finding intensity features
                mean_frames = np.nanmean(masked_frames,axis = tuple(masked_frames.ndim-1))
                std_frames = np.nanstd(masked_frames,axis = tuple(masked_frames.ndim-1))
                max_frames = np.nanmax(masked_frames,axis = tuple(masked_frames.ndim-1))
                min_frames = np.nanmin(masked_frames,axis = tuple(masked_frames.ndim-1))
                median_frames = np.nanmedian(masked_frames,axis = tuple(masked_frames.ndim-1))

                intensity_features_list = [mean_frames, std_frames, max_frames, min_frames, median_frames]

                # Adding features to annotations dictionary
                if 'user' not in annotations_json[0]['annotation']['elements'][nuc_idx]:
                    annotations_json[0]['annotation']['elements'][nuc_idx]['user'] = {}
                
                for feat_name, feat in zip(feature_list,intensity_features_list):
                    annotations_json[0]['annotation']['elements'][nuc_idx]['user'][feat_name] = feat.tolist()

            if isinstance(return_type,str):
                if return_type=='json':

                    return annotations_json, None
                
                elif return_type == 'csv':

                    nuclei_features_dataframe = self.format_df(annotations_json)

                    return None, nuclei_features_dataframe
            elif isinstance(return_type,list):
                
                # Plugin only expects one order anyways right?
                nuclei_features_dataframe = self.format_df(annotations_json)
                
                return annotations_json, nuclei_features_dataframe
            else:
                return annotations_json, None
        else:
            return None, None

    def format_df(self, annotations):

        # Taking an annotations object and pulling all the features out to return as a dataframe
        records_list = []
        for nuc_idx, nuc in annotations['annotation'][0]['elements']:

            nuc_dict = {
                'Name': f'Nucleus_{nuc_idx}'
            }

            # Getting bbox coordinates
            coordinates = np.array(nuc['points'])
            nuc_dict['Min_X'] = np.min(coordinates[:,0])
            nuc_dict['Min_Y'] = np.min(coordinates[:,1])
            nuc_dict['Max_X'] = np.max(coordinates[:,0])
            nuc_dict['Max_Y'] = np.max(coordinates[:,1])

            # Getting features
            for f in nuc['user']:
                if isinstance(nuc['user'][f],list):
                    for v_idx,v in enumerate(nuc['user'][f]):
                        nuc_dict[f'{f}_{v_idx}'] = v
                else:
                    nuc_dict[f] = nuc['user'][f]

            records_list.append(nuc_dict)

        nuc_df = pd.DataFrame.from_records(records_list)

        return nuc_df





