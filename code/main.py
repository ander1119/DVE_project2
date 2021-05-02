import argparse
import matplotlib.pyplot as plt
import cv2

import stitching
import feature

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_list', type=str, help='File containing image path and focal length')
    parser.add_argument('output_image', type=str, help='Path to the output image')

    args = parser.parse_args()
    images, focal_length = stitching.load_images(args.file_list)

    print('Warp image to cylindrical space...')
    projected_images = [stitching.cylindrical_projection(img, f) for img, f in zip(images, focal_length)]
    print('Done')

    offset_y = 0

    print('Computing panorama...')
    result_image = projected_images[0]
    for i in range(1, len(projected_images)):
        matched_pairs = feature.feature_match(projected_images[i], projected_images[i-1])
        offset = stitching.RANSEC(matched_pairs)
        print(f'Got offset {offset}')
        result_image = stitching.merge_two_image(projected_images[i], result_image, offset)
        offset_y += offset[0]

    result_image = stitching.end_to_end_align(result_image, offset_y)
    result_image = stitching.auto_crop(result_image)
    
    cv2.imwrite(args.output_image, result_image)
