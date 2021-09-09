import argparse
import random
import time
import warnings

import numpy as np

from neural_render.blender_render_utils.constants import SPLIT_, OUTPUT_IMAGE_DIR_, OUTPUT_SCENE_DIR_, \
    OUTPUT_SCENE_FILE_, \
    OUTPUT_QUESTION_FILE_, UP_TO_HERE_
from neural_render.blender_render_utils.helpers import render_image, initialize_paths

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(666)
np.random.seed(666)


def generate_images(max_images=2, batch_size=2, max_episodes=1, workers=1):
    global_generation_success_rate = 0

    ### MAIN LOOP ###
    episodes = 0
    start_time = time.time()
    while episodes < max_episodes:
        ### Move to Numpy Format
        key_light_jitter = [0.5, 0.5]
        fill_light_jitter = [0.5, 0.5]
        back_light_jitter = [0.5, 0.5]
        camera_jitter = [1, 1]
        per_image_shapes = [[0], [1]]
        per_image_colors = [[0], [1]]
        per_image_materials = [[0], [1]]
        per_image_sizes = [[0], [1]]

        per_image_x = [[0], [-0.3]]
        per_image_y = [[0], [-0.3]]
        per_image_theta = [[0], [0]]

        #### Render it ###
        attempted_images = render_image(key_light_jitter=key_light_jitter,
                                        fill_light_jitter=fill_light_jitter,
                                        back_light_jitter=back_light_jitter,
                                        camera_jitter=camera_jitter,
                                        per_image_shapes=per_image_shapes,
                                        per_image_colors=per_image_colors,
                                        per_image_materials=per_image_materials,
                                        per_image_sizes=per_image_sizes,
                                        per_image_x=per_image_x,
                                        per_image_y=per_image_y,
                                        per_image_theta=per_image_theta,
                                        num_images=batch_size,
                                        workers=workers,
                                        split=SPLIT_,
                                        assemble_after=False,
                                        start_idx=episodes * batch_size,
                                        )
        correct_images = [f[0] for f in attempted_images if f[1] == 1]
        global_generation_success_rate += len(correct_images)
        if global_generation_success_rate >= max_images:
            break
        episodes += 1
    end_time = time.time()
    ips = global_generation_success_rate / round(end_time - start_time, 2)
    duration = round(end_time - start_time, 2)
    print(f"Took {end_time - start_time} time for {global_generation_success_rate} images")
    print(f"Images per second {ips}")
    print(f"Generator Success Rate: {round(global_generation_success_rate / (max_episodes * batch_size), 2)}")
    return duration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpus', type=str, default=1)
    parser.add_argument('--output_image_dir', type=str, default=OUTPUT_IMAGE_DIR_)
    parser.add_argument('--output_scene_dir', type=str, default=OUTPUT_SCENE_DIR_)
    parser.add_argument('--output_scene_file', type=str, default=OUTPUT_SCENE_FILE_)
    parser.add_argument('--output_question_file', type=str, default=OUTPUT_QUESTION_FILE_)

    args = parser.parse_args()
    ngpus = str(args.ngpus)
    output_image_dir = str(args.output_image_dir)
    output_scene_dir = str(args.output_scene_dir)
    output_scene_file = str(args.output_scene_file)
    output_question_file = str(args.output_question_file)

    initialize_paths(output_scene_dir=OUTPUT_SCENE_DIR_, output_image_dir=OUTPUT_IMAGE_DIR_, up_to_here=UP_TO_HERE_)
    generate_images()
