import bpy
from mathutils import Vector
import math
import os
import os, sys, inspect
import pickle
import numpy as np
from typing import Tuple

# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])
)
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import bparser, butils


def get_theta_constraints():
    delta = np.radians(29.9)
    theta_inits = np.radians([90.0])
    theta_lows = theta_inits - delta
    theta_highs = theta_inits + delta
    return theta_lows, theta_highs


def get_phi_contraints():
    delta = np.radians(29.9)
    phi_inits = np.radians([135.0])
    phi_lows = phi_inits - delta
    phi_highs = phi_inits + delta
    return phi_lows, phi_highs


# def compute_rotations(tile_centers, focal, idx):
#     # add a batch dimension to the focal point
#     focal = np.array(focal)[np.newaxis, ...]
#     focal = np.tile(focal, (tile_centers.shape[0], 1))
#     # compute the direction vectors from the focal point to each tile center
#     # and normalize them
#     directions = focal - tile_centers
#     distances = np.linalg.norm(directions, axis=1)
#     theta = np.arccos(directions[:, 2] / distances)
#     phi = np.arctan2(directions[:, 1], directions[:, 0])
#     # if elements of phi > 0, wrap around by 2pi
#     if idx == 0:
#         phi[phi < 0] = phi[phi < 0] + 2 * np.pi
#     return theta, phi


def compute_rotations(tile_centers, focal, idx):

    # focal: first 3 idx: tx_position, the last 3 idx: focal point
    tx_position = focal[:3]
    focal = focal[3:]

    # add a batch dimension to the focal point and tx_position
    tx_position = np.array(tx_position)[np.newaxis, ...]
    tx_position = np.tile(tx_position, (tile_centers.shape[0], 1))
    focal = np.array(focal)[np.newaxis, ...]
    focal = np.tile(focal, (tile_centers.shape[0], 1))

    # compute normal vectors from the tile centers with the two points
    # focal and tx_position
    # focal: focal point, tx_position: transmitter position
    tile2focal = focal - tile_centers
    tile2focal = tile2focal / np.linalg.norm(tile2focal, axis=1, keepdims=True)
    tile2tx = tx_position - tile_centers
    tile2tx = tile2tx / np.linalg.norm(tile2tx, axis=1, keepdims=True)
    normals = 0.5 * (tile2focal + tile2tx)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # compute the direction vectors from the focal point to each tile center
    # and normalize them
    # directions = focal - tile_centers
    # distances = np.linalg.norm(directions, axis=1)
    theta = np.arccos(normals[:, 2])
    phi = np.arctan2(normals[:, 1], normals[:, 0])
    # if elements of phi < 0, wrap around by 2pi
    if idx == 0:
        phi[phi < 0] = phi[phi < 0] + 2 * np.pi
    return theta, phi


def compute_rot_angle(pt1: list, pt2: list) -> Tuple[float, float, float]:
    """Compute the rotation angles for vector pt1 to pt2."""
    # pt1: center, pt2: focal_pt -> focal_pt - center

    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    z = pt2[2] - pt1[2]

    return cartesian2spherical(x, y, z)


def cartesian2spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # theta: zenith angle (0, pi), phi: azimuthal angle (0, 2pi)
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    return r, theta, phi


def constrain_angles(angles, lows, highs):
    return np.clip(angles, lows, highs)


def generate_scene(args):
    """
    This function generates a scene in Blender using the provided arguments.

    A focal is in Cartesian coordinates
    Each reflector has a focal point
    All tiles of a reflector are pointing to the same focal point
    """
    # 1) Load focals
    with open(args.focals_path, "rb") as f:
        focals = pickle.load(f)

    # 2) Load constraints
    theta_lows, theta_highs = get_theta_constraints()
    phi_lows, phi_highs = get_phi_contraints()

    # 3) Get reflectors
    reflectors = {
        k: sorted(v.objects, key=lambda x: x.name)
        for k, v in bpy.data.collections.items()
        if "Reflector" in k
    }
    reflectors_names = sorted(reflectors.keys())

    # focals_viz = {
    #     k: sorted(v.objects, key=lambda x: x.name)
    #     for k, v in bpy.data.collections.items()
    #     if "Focal" in k
    # }
    # focals_viz_names = sorted(focals_viz.keys())
    # focals_viz = focals_viz[focals_viz_names[0]]

    # 4) Change orientation of reflector tiles using vectorized approach
    reflector = reflectors[reflectors_names[0]]
    # There are 9 groups of tiles in reflector, e.g. Group01.001, Group01.002, ..., Group01.009, Group02.001, ..., Group02.009
    # Each group has 9 tiles, e.g. Group01.001 has tiles:
    # Group01.001.Tile01, Group01.001.Tile02, ..., Group01
    tile_groups = [
        [tile for tile in reflector if tile.name.startswith(f"Group{str(i).zfill(2)}")]
        for i in range(1, 10)
    ]

    # for f_viz, focal in zip(focals_viz, focals):
    #     # Set the location of the focal point visualizer
    #     f_viz.location = Vector(focal[3:])
    #     # Set the name of the focal point visualizer
    #     # f_viz.name = f"Focal_{focal[3]:.2f}"

    for i, (tile_group, focal) in enumerate(zip(tile_groups, focals)):
        # Get the center of each tile in the group
        tile_centers = np.array([tile.location for tile in tile_group])
        theta, phi = compute_rotations(tile_centers, focal, 0)
        theta = constrain_angles(theta, theta_lows[0], theta_highs[0])
        phi = constrain_angles(phi, phi_lows[0], phi_highs[0])
        # Apply the rotations to each tile
        for tile, t, p in zip(tile_group, theta, phi):
            tile.rotation_euler = [0, t, p]

    # 5) Get other names
    obstacle_names = []
    for k, v in bpy.data.collections.items():
        if "Obstacle" in k:
            obstacle_names.append(k)

    # Save files without ceiling
    folder_dir = os.path.join(args.output_dir, f"idx")
    butils.mkdir_with_replacement(folder_dir)
    butils.save_mitsuba_xml(
        folder_dir, "scenee", [*reflectors_names, "Wall", "Floor", *obstacle_names]
    )
    # butils.save_mitsuba_xml(
    #     folder_dir,
    #     "scenee",
    #     [*reflectors_names, "Wall", "Floor", *obstacle_names, *focals_viz_names],
    # )

    # Save files with ceiling
    folder_dir = os.path.join(args.output_dir, f"ceiling_idx")
    butils.mkdir_with_replacement(folder_dir)
    butils.save_mitsuba_xml(
        folder_dir, "scenee", [*reflectors_names, "Wall", "Floor", "Ceiling", *obstacle_names]
    )


def main():
    args = create_argparser().parse_args()
    generate_scene(args)


def create_argparser() -> bparser.ArgumentParserForBlender:
    """Parses command line arguments."""
    parser = bparser.ArgumentParserForBlender()
    parser.add_argument("--scene_name", "-s", type=str, required=True)
    parser.add_argument("--focals_path", "-fp", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--index", type=str)
    return parser


if __name__ == "__main__":
    main()
