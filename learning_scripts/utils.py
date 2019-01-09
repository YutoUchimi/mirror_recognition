from __future__ import division

import numpy as np
import pascal3d
import PIL.Image
import skimage.restoration
import tqdm
import yaml

import tf


def load_off(filename):
    """Load OFF file.

    Parameters
    ----------
    filename: str
        OFF filename.
    """
    with open(filename, 'r') as f:
        first_line = f.readline()
        assert 'OFF' in first_line and 'COFF' not in first_line

        verts, faces = [], []
        n_verts, n_faces = None, None
        for line in f.readlines():
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if n_verts is None and n_faces is None:
                n_verts, n_faces, _ = map(int, line.split(' '))
            elif len(verts) < n_verts:
                verts.append([float(v) for v in line.split(' ')])
            else:
                faces.append([int(v) for v in line.split(' ')[1:]])
        verts = np.array(verts, dtype=np.float64)
        faces = np.array(faces, dtype=np.int64)

    return verts, faces


def transform_to_matrix(tf_file):
    f = open(tf_file, "r")
    transform = yaml.load(f)
    tx = transform['transform']['translation']['x']
    ty = transform['transform']['translation']['y']
    tz = transform['transform']['translation']['z']
    qx = transform['transform']['rotation']['x']
    qy = transform['transform']['rotation']['y']
    qz = transform['transform']['rotation']['z']
    qw = transform['transform']['rotation']['w']
    R_t = tf.transformations.translation_matrix((tx, ty, tz))
    R_t = np.asarray(R_t)
    R_r = tf.transformations.quaternion_matrix((qx, qy, qz, qw))
    R = np.hstack((R_r[:, :3], np.reshape(R_t[:, 3], (4, 1))))
    return R


def generate_depth(vertices, faces, height, width, R, K):
    vs = vertices
    vs = np.hstack((vs, np.ones((len(vs), 1))))
    vs_cam = np.dot(R, vs.T).T
    vertices_camframe = vs_cam[:, :3].T

    vs_cam = vs_cam[:, :3]
    I = np.dot(np.asarray(K).reshape(3, 3), vs_cam.T).T
    vs_2d = I[:, :2] / I[:, 2][:, np.newaxis]
    vs_2d = vs_2d[:, :2]
    vertices_2d = vs_2d

    vertices_camframe = vertices_camframe.transpose()
    polygons_z = np.abs(vertices_camframe[faces][:, :, 2])
    indices = np.argsort(polygons_z.max(axis=-1))

    # TODO(wkentaro): Get depth from raw data (Point Cloud?)
    depth = np.zeros((height, width), dtype=np.float64)
    depth.fill(np.inf)

    depth_obj = np.zeros((height, width), dtype=np.float64)
    depth_obj.fill(np.inf)
    mask_obj = np.zeros((height, width), dtype=bool)
    for face in tqdm.tqdm(faces[indices]):
        xy = vertices_2d[face].ravel().tolist()
        mask_pil = PIL.Image.new('L', (width, height), 0)
        PIL.ImageDraw.Draw(mask_pil).polygon(xy=xy, outline=1, fill=1)
        mask_poly = np.array(mask_pil).astype(bool)
        mask = np.bitwise_and(~mask_obj, mask_poly)
        mask_obj[mask] = True
        #
        if mask.sum() == 0:
            continue
        #
        ray1_xy = np.array(zip(*np.where(mask)))[:, ::-1]
        n_rays = len(ray1_xy)
        ray1_z = np.zeros((n_rays, 1), dtype=np.float64)
        ray1_xyz = np.hstack((ray1_xy, ray1_z))
        #
        ray0_z = np.ones((n_rays, 1), dtype=np.float64) * 10  # max depth: 10m
        ray0_xyz = np.hstack((ray1_xy, ray0_z))
        #
        tri0_xy = vertices_2d[face[0]]
        tri1_xy = vertices_2d[face[1]]
        tri2_xy = vertices_2d[face[2]]
        tri0_z = vertices_camframe[face[0]][2]
        tri1_z = vertices_camframe[face[1]][2]
        tri2_z = vertices_camframe[face[2]][2]
        tri0_xyz = np.hstack((tri0_xy, tri0_z))
        tri1_xyz = np.hstack((tri1_xy, tri1_z))
        tri2_xyz = np.hstack((tri2_xy, tri2_z))
        tri0_xyz = tri0_xyz.reshape(1, -1).repeat(n_rays, axis=0)
        tri1_xyz = tri1_xyz.reshape(1, -1).repeat(n_rays, axis=0)
        tri2_xyz = tri2_xyz.reshape(1, -1).repeat(n_rays, axis=0)
        #
        flags, intersection = pascal3d.utils.intersect3d_ray_triangle(
            ray0_xyz, ray1_xyz, tri0_xyz, tri1_xyz, tri2_xyz)
        for x, y, z in intersection[flags == 1]:
            depth_obj[int(y), int(x)] = z

    depth[mask_obj] = np.minimum(
        depth[mask_obj], depth_obj[mask_obj])
    depth[np.isinf(depth)] = np.nan

    inpaint_mask = mask_obj & np.isnan(depth)
    inpaint_depth = depth.copy()
    inpaint_depth[~mask_obj & ~inpaint_mask] = np.nanmean(depth)
    inpaint_depth = skimage.restoration.inpaint.inpaint_biharmonic(
        inpaint_depth, inpaint_mask, multichannel=False)
    inpaint_depth[inpaint_depth == 0] = np.nan
    depth = inpaint_depth.copy()

    return depth, mask_obj
