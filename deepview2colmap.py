from read_write_model import write_model, Camera, Image, Point3D
import argparse, os, re, torch
import numpy as np
from scipy.spatial.transform import Rotation
from timeit import default_timer as timer
import json
import cv2

def readCameraDeepview(path):
    cameras = {}
    images = {}
    with open(path, "r") as fi:
        js = json.load(fi)
        total_camera = len(js[0])
        for shot_number, shot_data in enumerate(js):
            for camera_id, cam_info in enumerate(shot_data):
                camera_id += 1 #force camera id start with 1
                if shot_number ==  0:
                    cameras[camera_id] = Camera(
                        id=camera_id,
                        model='PINHOLE',
                        width=int(cam_info['width']),
                        height=int(cam_info['height']),
                        params=[
                            cam_info['focal_length'], #fx
                            cam_info['focal_length'] * cam_info['pixel_aspect_ratio'], #fy
                            cam_info['principal_point'][0], # cx
                            cam_info['principal_point'][1] # cy
                        ]
                    )
            
                rotation, _ = cv2.Rodrigues(np.float32(cam_info['orientation']))
                #image_path = cam_info['relative_path']
                image_path = "shot{:03}_cam{:03}.jpg".format(shot_number,camera_id-1)
                image_id = total_camera * shot_number + camera_id
                rot = Rotation.from_matrix(rotation)
                qvec = rot.as_quat()
                images[image_id] = Image(
                    id=image_id,
                    qvec=np.array([qvec[3],qvec[0],qvec[1],qvec[2]]),
                    tvec=- np.matmul(rotation.T, np.array([cam_info['position']]).reshape(3)),
                    camera_id=camera_id,
                    name=image_path,
                    xys=np.array([]), 
                    point3D_ids=np.array([])
                )
    return cameras, images

def main(args):
    cameras, images = readCameraDeepview(args.input)
    print(len(cameras))
    exit()
    write_model(cameras,images,{},'output/','.bin')


def entry_point():
    start_timer = timer()
    parser = argparse.ArgumentParser(
        description='deeeparc2normalize.py - convert position of colmap from any position to object stay at -1 to 1')
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        #required=True,
        default='models.json',
        help='space dataset json file',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='output/',
        help='deeparc file output (default: \'output/\')')
    args = parser.parse_args()
    main(args)
    total_time = timer() - start_timer
    print('Finished in {:.2f} seconds'.format(total_time))
    print('output are write to {}'.format(os.path.abspath(args.output)))

if __name__ == "__main__":
    entry_point()