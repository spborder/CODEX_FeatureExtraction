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

from ctk_cli import CLIArgumentParser

sys.path.append('..')

import girder_client


def main(args):

    sys.stdout.flush()

    # Initialize girder client
    gc = girder_client.GirderClient(apiUrl = args.girderApiUrl)
    gc.setToken(args.girderToken)

    print('Input arguments:')
    for a in vars(args):
        print(f'{a}: {getattr(args,a)}')

    # Getting image information (image id)
    image_id = args.input_image
    image_info = gc.get(f'/item/{image_id}')
    print(f'Working on: {image_info["name"]}')

    # Copying it over to the plugin filesystem
    _ = gc.downloadItem(
            itemId = image_id,
            dest = '/',
            name = image_info['name']
        )
    print(f'Image copied successfully! {image_info["name"] in os.listdir("/")}')

    image_tiles_info = gc.get(f'/item/{image_id}/tiles')
    print(f'Image has {len(image_tiles_info["frames"])} Channels!')
    print(f'Image is {image_tiles_info["sizeY"]} x {image_tiles_info["sizeX"]}')

    





if __name__=='__main__':

    main(CLIArgumentParser().parse_args())

