import cv2
import numpy as np

img1 = cv2.imread("Question 2 and 3 Images/bikeL.png", 0)
img2 = cv2.imread("Question 2 and 3 Images/bikeR.png", 0)


#convert to uint8
img1 = cv2.convertScaleAbs(img1)
img2 = cv2.convertScaleAbs(img2)


#save the image
cv2.imwrite("Image1.png", img1)
cv2.imwrite("Image2.png", img2)



cam0 = np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]])
cam1 = np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]])
baseline = 177.288

image_shape  = (2008, 2988)

# Function to compute the disparity map using the block matching algorithm without using the OpenCV function

def compute_disparity_map(img1, img2, block_size=5, max_disp=64):
    """
    Compute the disparity map between two rectified images using the block matching algorithm.

    Args:
        img1: Left image.
        img2: Right image.
        block_size (int): Size of the block used for matching.
        max_disp (int): Maximum disparity value.

    Returns:
        Disparity map (numpy.ndarray)
    """
    h, w = img1.shape

    # Pad the images to handle boundary cases
    img1 = cv2.copyMakeBorder(img1, block_size, block_size, block_size, block_size, cv2.BORDER_CONSTANT)
    img2 = cv2.copyMakeBorder(img2, block_size, block_size, block_size, block_size, cv2.BORDER_CONSTANT)

    disparity_map = np.zeros((h, w), dtype=np.float32)

    for y in range(block_size, h + block_size):
        for x in range(block_size + max_disp, w + block_size):
            # Get the left and right blocks
            left_block = img1[y - block_size:y, x - block_size:x].astype(np.int32)
            best_disp = 0
            min_sad = float('inf')

            for disp in range(max_disp):
                right_block = img2[y - block_size:y, x - block_size - disp:x - disp].astype(np.int32)
                sad = np.sum(np.abs(left_block - right_block))

                if sad < min_sad:
                    min_sad = sad
                    best_disp = disp

            disparity_map[y - block_size, x - block_size - best_disp] = best_disp

    return disparity_map


# Compute the disparity map
disparity_map = compute_disparity_map(img1, img2, block_size=5, max_disp=64)

# calculate the depth map
depth_map = np.zeros_like(disparity_map, dtype=np.float32)
for y in range(depth_map.shape[0]):
    for x in range(depth_map.shape[1]):
        if disparity_map[y, x] > 0:
            depth_map[y, x] = (cam0[0, 0] * baseline) / disparity_map[y, x]


# function to compute the 3D point cloud
def compute_3d_point_cloud(disparity_map, cam0, baseline):
    """
    Computes the 3D point cloud from the disparity map.

    Args:
        disparity_map (numpy.ndarray): Disparity map.
        cam0 (numpy.ndarray): Intrinsic matrix of the left camera.
        baseline (float): Baseline.

    Returns:
        3D point cloud (numpy.ndarray)
    """
    h, w = disparity_map.shape
    f = cam0[0, 0]

    point_cloud = np.zeros((h, w, 3), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            if disparity_map[y, x] > 0:
                Z = (f * baseline) / disparity_map[y, x]
                X = (x - cam0[0, 2]) * Z / f
                Y = (y - cam0[1, 2]) * Z / f
                point_cloud[y, x] = [X, Y, Z]

    return point_cloud



# Compute the 3D point cloud
point_cloud = compute_3d_point_cloud(disparity_map, cam0, baseline)


# Save the disparity map
cv2.imwrite("disparity_map.png", disparity_map)

# Save the depth map
cv2.imwrite("depth_map.png", depth_map)

# Save the point cloud
np.save("point_cloud.npy", point_cloud)
#sparity map as csv
np.savetxt("disparity_map.csv", disparity_map, delimiter=",")
#depth map as csv
np.savetxt("depth_map.csv", depth_map, delimiter=",")


from mpl_toolkits.mplot3d import Axes3D
# Load the point cloud
point_cloud = np.load("point_cloud.npy")

# Plot the 3D point cloud from xy axis perspective 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, :, 0], point_cloud[:, :, 1], point_cloud[:, :, 2], c='b', marker='o', s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

