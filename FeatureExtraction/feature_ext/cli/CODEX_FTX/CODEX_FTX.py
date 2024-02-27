"""

CODEX nuclei segmentation and feature extraction plugin

    - Select frame to use for nuclei segmentation
    - Iterate through tiles? (Remove border nuclei)
    - Segment nuclei using user-specified parameters
        - Threshold, minimum size, something for splitting adjacent nuclei
    - Create nuclei annotations
    - Iterate through all frames
    - Extract frame intensity statistics
        - Mean, median, standard deviation, maximum, minimum, etc.
    - Add to annotation user metadata
    - Post entire thing to slide annotations?

Find some way to leverage GPUs or something to make this more efficient.
Can annotations be appended to or just posted all at once?

"""

import os
import sys
from math import ceil, floor
import numpy as np
import json
import pandas as pd

from ctk_cli import CLIArgumentParser

sys.path.append('..')
from CODEX_Extractor import CODEXtractor

import girder_client


class Patches:
    def __init__(self,
                 image_id:str,
                 patch_size:int,
                 region,
                 gc):
        
        self.image_id = image_id
        self.patch_size = patch_size
        self.region = region
        self.gc = gc

        # Getting image metadata
        self.image_info = self.gc.get(f'/item/{self.image_id}')
        self.image_metadata = self.gc.get(f'/item/{self.image_id}/tiles')

        if len(self.region) is None:
            self.region = [
                0,
                0,
                self.image_metadata['sizeX'],
                self.image_metadata['sizeY']
            ]

        self.regions_list = self.get_regions_list()
        print(f'These are the regions: {self.regions_list}')

    def get_regions_list(self):
        
        # Defining list of all possible non-overlapping regions within the selected region
        region_height = self.region[3]
        region_width = self.region[2]

        if region_height <= self.patch_size and region_width <= self.patch_size:
            region_list = [[int(self.region[0]),int(self.region[1]),int(self.region[0]+region_width),int(self.region[1]+region_height)]]

            return region_list

        else:
            n_patch_x = ceil(region_width/self.patch_size)
            n_patch_y = ceil(region_height/self.patch_size)

            patch_regions_list = []

            for x in range(0,n_patch_x):
                for y in range(0,n_patch_y):

                    # Finding patch start coordinates
                    patch_start_x = np.minimum(self.region[0]+(x*self.patch_size),(self.region[0]+region_width)-(region_width%self.patch_size))
                    patch_start_y = np.minimum(self.region[1]+(y*self.patch_size),(self.region[1]+region_height)-(region_height%self.patch_size))

                    # Finding patch end coordinates
                    patch_end_x = np.minimum(patch_start_x+self.patch_size, self.region[0]+region_width)
                    patch_end_y = np.minimum(patch_start_y+self.patch_size,self.region[1]+ region_height)

                    patch_regions_list.append([patch_start_x,patch_start_y,patch_end_x,patch_end_y])
            
            return patch_regions_list

    def __iter__(self):

        self.patch_idx = -1

        return self
    
    def __next__(self):

        self.patch_idx+=1
        if self.patch_idx<len(self.regions_list):
            return self.regions_list[self.patch_idx]
        else:
            raise StopIteration


def main(args):

    sys.stdout.flush()

    # Initialize girder client
    gc = girder_client.GirderClient(apiUrl = args.girderApiUrl)
    gc.setToken(args.girderToken)

    print('Input arguments:')
    for a in vars(args):
        print(f'{a}: {getattr(args,a)}')

    # Getting image information (image id)
    image_id = gc.get(f'/file/{args.input_image}')['itemId']
    image_info = gc.get(f'/item/{image_id}')
    print(f'Working on: {image_info["name"]}')

    # Copying it over to the plugin filesystem
    _ = gc.downloadFile(
            fileId = args.input_image,
            path = f'/{image_info["name"]}',
        )
    print(f'Image copied successfully! {image_info["name"] in os.listdir("/")}')

    image_tiles_info = gc.get(f'/item/{image_id}/tiles')
    print(f'Image has {len(image_tiles_info["frames"])} Channels!')
    print(f'Image is {image_tiles_info["sizeY"]} x {image_tiles_info["sizeX"]}')

    # Creating patch iterator 
    patch_maker = Patches(
        image_id = image_id,
        patch_size = args.patch_size,
        region = args.input_region,
        gc = gc
    )

    patch_maker = iter(patch_maker)

    # Initializing feature extractor
    feature_maker = CODEXtractor(
        image_id = image_id,
        seg_params = {
            'frame':args.nuclei_frame,
            'threshold':args.threshold_nuclei,
            'min_size': args.minsize_nuclei,
            'cyto_pixels': args.cyto_pixels
            },
        gc = gc
    )

    # Getting the return type(s) sorted
    if ',' in args.return_type:
        return_type = args.return_type.split(',')
        
        if 'json' in return_type:
            return_annotations = True
        else:
            return_annotations = False
        
        if 'csv' in return_type:
            return_csv = True
        else:
            return_csv = False

    else:
        return_type = args.return_type

        if return_type == 'json':
            return_annotations = True
        else:
            return_annotations = False

        if return_type == 'csv':
            return_csv = True
        else:
            return_csv = False
    
    if return_annotations:
        # Initializing empty annotations object
        all_nuc_annotations = [{
            'annotation': {
                'name': 'CODEX Nuclei',
                'attributes': {},
                'elements': []
            }
        }]

    if return_csv:
        all_nuc_df = pd.DataFrame()

    more_patches = True
    while more_patches:
        try:
            # Getting the next patch region
            next_region = next(patch_maker)
            print(f'On patch: {patch_maker.patch_idx+1} of {len(patch_maker.regions_list)}')

            # Getting features and annotations within that region
            region_annotations, region_df = feature_maker.get_intensity_features(
                region_coords = next_region,
                return_type = return_type
            )

            print(f'return_annotations: {return_annotations}')
            print(f'Found: {len(region_annotations[0]["annotation"]["elements"])} Nuclei')

            if return_annotations:
                if not region_annotations is None:
                    # Adding to total annotations object
                    all_nuc_annotations[0]['annotation']['elements'].extend(region_annotations[0]['annotation']['elements'])

            if return_csv:
                if not region_df is None:
                    # Adding to all_nuc_df
                    if all_nuc_df.empty:
                        all_nuc_df = region_df
                    else:
                        all_nuc_df = pd.concat([all_nuc_df,region_df],axis = 0, ignore_index = True)


        except StopIteration:
            more_patches = False


    if return_annotations:
        # Posting annotations to item
        gc.post(f'/annotation/item/{image_id}?token={args.girderToken}',
                data = json.dumps(all_nuc_annotations),
                headers = {
                    'X-HTTP-Method': 'POST',
                    'Content-Type': 'application/json'
                    }
                )

    if return_csv:
        # Adding csv to image item files
        all_nuc_df.to_csv('/Nuclei_features.csv')
        gc.uploadFileToItem(
            itemId = image_id,
            filepath = '/Nuclei_features.csv'
        )



if __name__=='__main__':

    main(CLIArgumentParser().parse_args())

