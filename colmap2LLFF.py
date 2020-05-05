from read_write_model import write_model, read_model, Camera, Image, Point3D
import argparse, os, re, torch
import numpy as np
from scipy.spatial.transform import Rotation
from timeit import default_timer as timer
import json
import cv2

def save_image_preserve_path(image,image_name,args):
    split_path = image_name.split('/')
    if len(split_path) > 1:
        dir_path = os.path.join(args.image_output,'/'.join(split_path[:-1]))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    cv2.imwrite(os.path.join(args.image_output,image_name),image)

def get_short_number(name):
    return int(name.split('_')[2].split('.')[0])

def get_image_name(shot_number, camera_id):
    image_path = "shot{:03}_cam{:03}.jpg".format(shot_number,camera_id-1)
    return image_path

def get_smallest_height(args):
    cam_dirs = filter(lambda x: os.path.isdir(os.path.join(args.image,x)), os.listdir(args.image))
    heights = []
    for cam_dir in cam_dirs:
        cam_path = os.path.join(args.image,cam_dir)
        files = os.listdir(cam_path)
        image = cv2.imread(os.path.join(cam_path,files[0]))
        heights.append(image.shape[0])
    return min(heights)


def crop_image(cameras, images, args):
    new_cameras = {}
    new_images = {}
    smallest_height = get_smallest_height(args)
    for image_id in images:
        image_name = images[image_id][4]
        camera_id = images[image_id][3]
        image_path = os.path.join(args.image,image_name)
        image = cv2.imread(image_path)
        height, width, channel = image.shape
        if camera_id not in new_cameras:
            if cameras[camera_id][1] != 'PINHOLE':
                raise RuntimeError('This camera model isnt support yet')
            # PINHOLE camera
            cam_info = cameras[camera_id]
            param_info = cameras[camera_id][4]
            params = [
                param_info[0] * width/cam_info[2], #fx
                param_info[1] * smallest_height/cam_info[3], #fy
                param_info[2] * width/cam_info[2], #cx
                param_info[3] * smallest_height/cam_info[3] #cy
            ]
            new_cameras[camera_id] = Camera(
                id=camera_id,
                model=cameras[camera_id][1],
                width=width,
                height=smallest_height,
                params=params
            )
        shift_height = (height - smallest_height)//2
        image = image[shift_height:shift_height+smallest_height, :, :]
        short_number = get_short_number(image_name)
        new_image_name = get_image_name(short_number,camera_id)
        cv2.imwrite(os.path.join(args.image_output,new_image_name),image)

        new_images[image_id] = Image(
            id=image_id,
            qvec=images[image_id][1], 
            tvec=images[image_id][2],
            camera_id=camera_id, 
            name=new_image_name,
            xys=images[image_id][5], 
            point3D_ids=images[image_id][6]
        )
        
    return new_cameras, images

def main(args):
    cameras, images, points3D = read_model(args.input, '.bin')
    if not os.path.exists(args.image_output):
        os.makedirs(args.image_output)
    if not os.path.exists(args.model_output):
        os.makedirs(args.model_output)
    new_cameras, new_images = crop_image(cameras,images,args)
    write_model(new_cameras,new_images, points3D, args.model_output,'.bin')


def entry_point():
    start_timer = timer()
    parser = argparse.ArgumentParser(
        description='colmap2LLFF')
    parser.add_argument(
        '--input',
        type=str,
        #required=True,
        default='C:\\Datasets\\spaces_dataset\\data\\output_800\\scene_000\\sparse\\undistort_group',
        help='',
    )
    parser.add_argument(
        '--image',
        type=str,
        #required=True,
        default='C:\\Datasets\\spaces_dataset\\data\\800\\scene_000',
        help='',
    )
    parser.add_argument(
        '--image-output',
        type=str,
        default='images/',
        help='')
    parser.add_argument(
        '--model-output',
        type=str,
        default='output/',
        help='')
    args = parser.parse_args()
    main(args)
    total_time = timer() - start_timer
    print('Finished in {:.2f} seconds'.format(total_time))
    print('model are write to {}'.format(os.path.abspath(args.model_output)))
    print('images are save to {}'.format(os.path.abspath(args.image_output)))

if __name__ == "__main__":
    entry_point()