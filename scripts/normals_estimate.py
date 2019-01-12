
import math, os, sys, random, glob, itertools
import parse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

from PIL import Image
from multiprocessing import Pool
from skimage.filters import gaussian

from logger import Logger, VisdomLogger
from utils import *


TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'
def cam_read(filename):
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def normalize(v, axis=0):
    return v/np.linalg.norm(v, axis=axis, keepdims=True)

def load_depth_map_in_m(file_name):
    image = depth_read(file_name)
    print (image.min(), image.max())
    return image

def pixel_to_ray(pixel,camera,pixel_width=320,pixel_height=240):
    x, y = pixel
    hfov = 2*np.arctan(pixel_height/(2*camera[0, 0]))
    vfov = 2*np.arctan(pixel_width/(2*camera[1, 1]))
    x_vect = math.tan(hfov/2.0) * ((2.0 * ((x+0.5)/pixel_width)) - 1.0)
    y_vect = math.tan(vfov/2.0) * ((2.0 * ((y+0.5)/pixel_height)) - 1.0)
    return (x_vect,y_vect,1.0)

def normalised_pixel_to_ray_array(camera,width=320,height=240):
    pixel_to_ray_array = np.zeros((height,width,3))
    for y in range(height):
        for x in range(width):
            pixel_to_ray_array[y,x] = normalize(np.array(pixel_to_ray((x,y),camera,pixel_height=height,pixel_width=width)))
    return pixel_to_ray_array

def points_in_camera_coords(depth_map,pixel_to_ray_array):
    # depth_map = gaussian(depth_map, sigma=2)
    assert depth_map.shape[0] == pixel_to_ray_array.shape[0]
    assert depth_map.shape[1] == pixel_to_ray_array.shape[1]
    assert len(depth_map.shape) == 2
    assert pixel_to_ray_array.shape[2] == 3
    camera_relative_xyz = np.ones((depth_map.shape[0],depth_map.shape[1],4))
    for i in range(3):
        camera_relative_xyz[:,:,i] = depth_map * pixel_to_ray_array[:,:,i]
    
    return camera_relative_xyz

# A very simple and slow function to calculate the surface normals from 3D points from
# a reprojected depth map. A better method would be to fit a local plane to a set of 
# surrounding points with outlier rejection such as RANSAC.  Such as done here:
# http://cs.nyu.edu/~silberman/projects/indoor_scene_seg_sup.html
def surface_normal(points, pixel_width=320, pixel_height=240):
    # These lookups denote y,x offsets from the anchor point for 8 surrounding
    # directions from the anchor A depicted below.
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 | A | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    d = 2
    lookups = {0:(-d,0),1:(-d,d),2:(0,d),3:(d,d),4:(d,0),5:(d,-d),6:(0,-d),7:(-d,-d)}
    
    diff = np.zeros((points.shape[0], points.shape[1], 8))
    cross = np.zeros((3, points.shape[0], points.shape[1], 8))
    for k in range(8):
        point1 = points[:, :, :3]
        point2 = points.take(range(lookups[k][0], lookups[k][0]+points.shape[0]), axis=0, mode='wrap')\
            .take(range(lookups[k][1], lookups[k][1]+points.shape[1]), axis=1, mode='wrap')[:, :, :3]

        point3 = points.take(range(lookups[(k+2)%8][0], lookups[(k+2)%8][0]+points.shape[0]), axis=0, mode='wrap')\
            .take(range(lookups[(k+2)%8][1], lookups[(k+2)%8][1]+points.shape[1]), axis=1, mode='wrap')[:, :, :3]
        diff[:, :, k] = np.linalg.norm(point2 - point1, axis=2) + np.linalg.norm(point3 - point1, axis=2)
        cross_product = normalize(np.cross(point2 - point1, point3 - point1), axis=2)
        cross[:, :, :, k] = cross_product.transpose((2, 0, 1))
    
    normal = np.zeros((points.shape[0], points.shape[1], 3))
    for i, j in itertools.product(range(points.shape[0]), range( points.shape[1])):
        normal[i, j, :] = cross[:, i, j, np.argmin(diff[i, j, :])]
    return normal


def normal_from_depth(file_name, cam_file=None, image_file=None):
    
    depth_map = load_depth_map_in_m(file_name)

    if cam_file is None:
        print (file_name)
        result = parse.parse("mount/sintel/training/{task_dir}/{building}/{task_val}_{view}.dpt", file_name)
        building, view = (result["building"], result["view"])
        cam_file = f"mount/sintel/training/camdata_left/{building}/frame_{view}.cam"

    camera, _ = cam_read(cam_file)
    print (camera)

    cached_pixel_to_ray_array = normalised_pixel_to_ray_array(camera, width=1024, height=436)

    points_in_camera = points_in_camera_coords(depth_map, cached_pixel_to_ray_array)
    surface_normals = surface_normal(points_in_camera, pixel_width=depth_map.shape[1], pixel_height=depth_map.shape[0])
    surface_normals = 1.0 - surface_normals*0.5 - 0.5
    return file_name, surface_normals





if __name__ == "__main__":
    files = glob.glob("mount/sintel/training/depth/*/frame*.dpt")
    depth_map = load_depth_map_in_m(files[0])
    normals_list = []

    logger = VisdomLogger("train", env=JOB)

    with Pool() as pool:
        for i, (file, normals) in enumerate(
            pool.imap_unordered(normal_from_depth, files)
        ):

            filename = "normal" + file.split('/')[-1][5:][:-4] + ".png"
            filename = file[:-len(filename)] + "/" + filename
            print (i, len(files), filename)
            plt.imsave(filename, normals)
            normals_list.append(normals)
            print (len(normals_list))

    # normals_list = torch.FloatTensor(np.array(normals_list).astype(np.float32))
    # normals_list = normals_list.permute((0, 3, 1, 2))
    # logger.images(normals_list, "normals", resize=256)



