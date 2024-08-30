import numpy as np

# How doe it work:
"""
input:
    angle: [degree]:
        positive: it moves clockwise in plane (x,z) with respect to the center.
        negative: it moves counterclockwise in plane (x,z) with respect to the center.
    
    center_x, center_z: coordinates of the center (center of line between the shoulder).

    point_x, point_z: coordinates of the point that we want to rotate.

    it return only the x coordinated since we don't need the z and y it's the same (don't change).

"""

def rotate_point_depth(angle, center_x, center_z, point_x, point_z):
    # Since i want to use a matrix for rotation, i would like to have
    # the point coordinate set as the center point is the origin of
    # the new center of the system.
    new_point_x = point_x - center_x
    new_point_z = point_z - center_z

    angle_rad = (angle * np.pi) / 180

    rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
    
    rotated = np.matmul(rot_mat, np.array([new_point_x, new_point_z]))

    # Now i need to return only x coordinate of the point.
    return int(rotated[0] + center_x)

if __name__ == '__main__':
    # Testing.
    ret = rotate_point_depth(43.6, 100, 100, 150, 80)
    print(f"Should be around: 150, it's: {ret}")
    ret = rotate_point_depth(-43.6, 100, 100, 150, 120)
    print(f"Should be around: 150, it's: {ret}")
    ret = rotate_point_depth(43.6, 100, 100, 120, 150)
    print(f"Should be around: 80, it's: {ret}")
    ret = rotate_point_depth(-43.6, 100, 100, 80, 150)
    print(f"Should be around: 120, it's: {ret}")