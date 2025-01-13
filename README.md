# Harris-Corner-Detection-and-Stereo-3D-Reconstruction

## Part 1: Harris Corner Detection (Scratch)

Harris corner detection method (from scratch) is applied to given images to detect corners. Results are then compared to those obtained using the OpenCV (cv2) library.

Shown below are some of the results:

![harris_corner_detection_results1](https://github.com/user-attachments/assets/3cb8e838-b6d4-4e7d-bed7-4e74a59196a9)

![harris_corner_detection_results2](https://github.com/user-attachments/assets/0016f406-7fc4-4da4-b945-d6e85d647014)

![harris_corner_detection_results4](https://github.com/user-attachments/assets/3eec2a39-a2cc-4add-9adf-8981776e82a4)

![harris_corner_detection_results8](https://github.com/user-attachments/assets/266a199d-0f91-4aad-8dee-e7658ac07901)

## Part 2: Stereo 3D Reconstruction to find disparity map, depth map and 3D point cloud

Two stereo images I1 (“bikeL.png”) and I2 (“bikeR.png”) of a static scene captured from a stereo camera with the given intrinsic matrices of both cameras are available (in the file (“bike.txt”)). Stereo 3D reconstruction algorithm is used to find the disparity map, depth map (depth map at each pixel), and 3D point cloud representation of the underlying scene.

![disparity_map](https://github.com/user-attachments/assets/54d8ac81-2731-4a42-b6ca-2cf188ccce34)

Depth Map:
![depth_map](https://github.com/user-attachments/assets/62033df8-ce1e-4a7a-bd6b-1e96ad0606bc)

3D Point Cloud:
![3D_PointCloud](https://github.com/user-attachments/assets/78047950-bc7a-455c-b3df-27fb5f794739)
