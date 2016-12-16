# Horn-Schunck Optical Flow
We will implement Horn-Schunk's optical flow algorithm in Matlab and C++. 
Our goal is to learn the algorithm and compare the performance metrics (time, memory, and etc.) of the two implementations.

## How to Run Matlab Implementation
First make sure the folder containing the matlab code is added to path. Then simply run the following command:

  `[U, V, elasped_time] = optical_flow();`

This runs the Horn-Schunck optical flow algorithm with default parameters. UI will pop up to let you select the two images.

Alternatively, you can also specify the parameters using:

  `[U, V, elasped_time] = optical_flow(file1, file2, num_it, avg_window, alpha);`

- file1 and file2 are the filenames of the two images.

- num_it is the number of iterations.

- avg_window is the number of columns (also equal to the number of rows) of the averaging kernel.

- alpha is the regularization constant.

## How to Run C++ Implementation
First, download the OpenCV library from http://opencv.org/downloads.html. Particularly, the optical flow code developed for this repository was created under the following environment:

- Windows 8, 64-bit
- Microsoft Visual Studio 2015
- OpenCV 3.0

To emulate this exact environment, watch the following video (thanks to the user nurimbet!): https://www.youtube.com/watch?v=l4372qtZ4dc

For other environments, refer to the documentation provided by OpenCV: http://opencv.org/documentation.html

Alternatively, for a Windows environment, use the executable and provided .dll file from the `cpp_version` folder to run the optical flow analysis in C++ without the aforementioned dependencies.

**Usage:** `optical_flow.exe IMAGE1_FILEPATH IMAGE2_FILEPATH`
