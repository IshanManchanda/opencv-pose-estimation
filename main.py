import glob
import math
import os

import cv2
import numpy as np
from cv2 import aruco

from calibration import get_camera_parameters


def get_angles(rvec):
	# Convert rotation vector to rotation matrix
	# Second return value is the Jacobian matrix, matrix of partial derivatives
	# of the output array components with respect to the input array components
	r_mat, _ = cv2.Rodrigues(rvec)

	# REVIEW: Look into "height of the camera"
	# Can compute position of camera in the world frame as:
	# cam_pos = -r_mat.T * tvec

	# Check for singularity
	sin_x = math.sqrt(r_mat[2, 0] * r_mat[2, 0] + r_mat[2, 1] * r_mat[2, 1])
	singularity = sin_x < 1e-6
	if not singularity:
		# Formulae to decompose rotation matrix into yaw, pitch, and roll
		z1 = math.atan2(r_mat[2, 0], r_mat[2, 1])  # around z1-axis
		x = math.atan2(sin_x, r_mat[2, 2])  # around x-axis
		z2 = math.atan2(r_mat[0, 2], -r_mat[1, 2])  # around z2-axis

	# Near the singularity, the above method becomes numerically unstable
	# This is called Gimbal Lock, when 2 of the axes having a
	# parallel configuration reduces one degree of freedom
	else:
		z1 = 0  # around z1-axis
		x = math.atan2(sin_x, r_mat[2, 2])  # around x-axis
		z2 = 0  # around z2-axis

	angles = -180 * np.array([z1, x, z2]) / np.pi
	return angles


def main():
	# Get paths to all images in the input directory
	base_path = os.path.dirname(os.path.abspath(__file__))
	input_path = os.path.join(base_path, 'data/*.png')
	# We use the sort to prevent string sorting of 10 before 2 etc.
	img_paths = sorted(glob.glob(input_path), key=len)

	# Get the Aruco dictionary and default parameter object
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters = aruco.DetectorParameters_create()

	# The printed tag is about 52.8 mm = 0.0528 m
	marker_size = 0.0528

	# Compute intrinsic parameters using camera calibration
	mtx, dist = get_camera_parameters()

	# Iterate over all input images
	for i, path in enumerate(img_paths):
		img = cv2.imread(path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Detect markers in the image
		corners, ids, _ = aruco.detectMarkers(
			gray, aruco_dict, parameters=parameters
		)

		# Estimate pose using the marker and intrinsic parameters
		# Last return value is a list of object points for the corners
		rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
			corners, marker_size, mtx, dist
		)

		# Compute distance using the translation vector
		print(f"img {i + 1} Distance:", np.sqrt(np.sum(np.square(tvec))))

		# Compute yaw, pitch, roll angles from the rotation vector
		yaw, pitch, roll = get_angles(rvec)
		print(f"img {i + 1} Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")

		# Make a copy of the image and draw the axis on it
		img_axis = img.copy()
		aruco.drawAxis(img_axis, mtx, dist, rvec, tvec, 0.1)
		# Save the image to the out directory
		out_path = os.path.join(base_path, f'out/out_{i}.png')
		cv2.imwrite(out_path, img_axis)


if __name__ == '__main__':
	main()
