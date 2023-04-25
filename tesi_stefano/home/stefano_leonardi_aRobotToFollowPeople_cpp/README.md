# A Robot To Follow People - C++ version
> This repository is a re-implementation of the repository [aRobotToFollowPeople](https://github.com/leopold-lll/aRobotToFollowPeople) **originally implemented in python**. The goal is to **re-write the code in C++** in order to interact with other services and also to improve the performances a little bit.

This new repository is dedicated to an extension of the master thesis research project of **Stefano Leonardi** at the **[University of Trento](https://www.unitn.it/en)** for the **[Dolomiti Robotics](https://dolomitirobotics.it/)** group.

##### Cross platform application:
The C++ implementation use **CMake to compile on multiple OS**: both on Windows and various Linux distributions. In addition, the code also use a **Dockerfile to work on raw devices** where the neccessary libraries, such as OpenCV, are not installed.
The basic template to work with CMake, Docker and OpenCV, that were used as base of this implementation, can be found at the [cMake_template repository](https://github.com/leopold-lll/cMake_template).

### The goal
The goal of this code is to allow a robot equipped with an RGB camera to detect, identify and track a person in a real-time video. Then, with the use of a LIDAR sensor follows it across time through a real environment.

### Necessary  material
The repository contains only the code of the software, but the entire project requires additional elements to work:
- The DNN (Deep Neural Network) [pre-trained models](https://drive.google.com/drive/folders/1NIsFhys1TO4IEbt0ZBErVAGFfm2TB14L?usp=sharing).
- The [database of images](https://drive.google.com/drive/folders/1UG_BCHDNZywuIp5mIVAFgByNDbOBzKWA?usp=sharing) internally used as samples when there are no people in the field of view of the robot.
- If GOTURN tracker is used the relative caffe models should be downloaded and placed in the root folder of the project OpenCV does not allow to place these files into a different location... The [prototxt](https://drive.google.com/file/d/1vSmrPjZBM-UAhVSLRxpj_mrlRPC4-AsG/view?usp=sharing) and [caffemodel](https://drive.google.com/file/d/18I5TA17-tdCfSM9SwCZeGivHiYethbPu/view?usp=sharing) files.

And, **optionally** also:
- A [set of input samples](https://drive.google.com/drive/folders/1v_NtzNaYFYeP-5k7Mzwyy3rw1KzmDXAI?usp=sharing) recorded to test the potentialities of the software.

These folders should be placed in the root location of the project.

### References
A quick and complete view of the project can be done with the use of:
- The complete documentation of the code. Available at the relative path: "./docs/html/index.html"
- A [demo](https://drive.google.com/file/d/1s_sXa-Q7-MVhQVobPWRdU7K5oFsKyt4N/view?usp=sharing) of the **python version** of the working software executed on an Intel Core i5 CPU.
- The [final dissertation](https://github.com/leopold-lll/thesis_aRobotToFollowPeople/blob/master/main.pdf) containing all the details and the implemental choices that have been done.
The dissertation is titled: *"Integration of multiple deep learning algorithms for real-time tracking of a person in complex scenarios"*.
- The powerpoint [presentation](https://drive.google.com/file/d/1jJQ9YGHTVK5UrLhepHpZLqO4kwQCIXDb/view?usp=sharing) of the overall project.
- The [recorded oral discussion](https://drive.google.com/file/d/1vLVMeXBxDt49J976ht0A8TKUEt3Bqjcf/view?usp=sharing) of the presentation.
