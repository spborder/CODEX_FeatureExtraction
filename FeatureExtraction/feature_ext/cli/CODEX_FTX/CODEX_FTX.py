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

    def get_regions_list(self):
        
        # Defining list of all possible non-overlapping regions within the selected region

        region_height = self.region[3]-self.region[1]
        region_width = self.region[2]-self.region[0]

        if region_height <= self.patch_size and region_width <= self.patch_size:
            return [self.region]
        else:
            n_patch_x = ceil(region_width/self.patch_size)
            n_patch_y = ceil(region_height/self.patch_size)

            patch_regions_list = []

            for x in range(0,n_patch_x):
                for y in range(0,n_patch_y):

                    # Finding patch start coordinates
                    patch_start_x = np.minimum(self.region[0]+(x*self.patch_size),self.region[2]-(self.region[2]%self.patch_size))
                    patch_start_y = np.minimum(self.region[0]+(y*self.patch_size),self.region[3]-(self.region[3]%self.patch_size))

                    # Finding patch end coordinates
                    patch_end_x = np.minimum(patch_start_x+self.patch_size, self.region[2])
                    patch_end_y = np.minimum(patch_start_y+self.patch_size,self.region[3])

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
            itemId = args.input_image,
            dest = '/',
            name = image_info['name']
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
        region = args.input_region,
        seg_params = {'frame':args.nuclei_frame,'threshold':args.threshold_nuclei,'min_size': args.minsize_nuclei},
        gc = gc
    )

    # Initializing empty annotations object
    all_nuc_annotations = [{
        'annotation': {
            'name': 'CODEX Nuclei',
            'attributes': {},
            'elements': [],
            'user': {
                'segmentation_parameters': {
                    'frame': args.nuclei_frame,
                    'threshold': args.threshold_nuclei,
                    'min_size': args.minsize_nuclei
                }
            }
        }
    }]

    more_patches = True
    while more_patches:
        try:
            print(f'On patch: {patch_maker.patch_idx} of {len(patch_maker.regions_list)-1}')
            # Getting the next patch region
            next_region = next(patch_maker)
            # Getting features and annotations within that region
            region_annotations = feature_maker.get_intensity_features(
                region_coords = next_region,
                return_type = 'annotation'
            )

            # Adding to total annotations object
            all_nuc_annotations[0]['annotation']['elements'].extend(region_annotations[0]['annotation']['elements'])
        except StopIteration:
            more_patches = False


    # Posting annotations to item
    gc.post(f'/annotation/item/{image_id}?token={args.girderToken}',
            data = json.dumps(all_nuc_annotations),
            headers = {
                'X-HTTP-Method': 'POST',
                'Content-Type': 'application/json'
                }
            )



if __name__=='__main__':

    main(CLIArgumentParser().parse_args())

