/** 
 *  @file   videoIO.hh
 *  @brief  A file containing the implementation of the StoreVideo and LoadVideo classes.
 *  @author Stefano Leonardi
 ***********************************************/
#pragma once //force the compiler to load this file only once (the same of #define UNIQUE_NAME)

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "utils.hh"

using std::cout;
using std::endl;
using std::string;
using cv::Mat;
using cv::VideoWriter;
using cv::VideoCapture;
using cv::waitKey;

/*! The StoreVideo class is designed to manage the interface for save a video on the local OS and optionally watch it. By taking one frame at a time. The video can be show to the user that could also stop the reproduction.
*/
class StoreVideo{
    private:
        string dest; //destination
        bool show;
        int fps;
        int fourcc;
        // std::optional<VideoWriter> writer;   // c++ 17 version
        VideoWriter writer;                     // c++ 11 version
        bool toBeInitialized;

    public:
        /** \brief The constructor method of the class.
         *  \param[in] dest The path to the location where to save the generated video. The path is without extension (it will be automatically added).
         *  \param[in] show A flag that says if the passed frames should be also show to the user or not. 
         *  \param[in] fps The fps used to mount the video with the given frames. (less than 5/10 fps it is not suggested due to strange behaviors)
         */
        StoreVideo(const string & dest="out", bool show=false, int fps=10);

        /** \brief Add a single frame to the video.
         *  \param[in] frame The frame to be added.
         *  \return A flag, true if the user ask to stop the video, false otherwise.
         */
        bool addFrame(const Mat & frame);
        
        /** \brief Release all the resources used for the internal VideoWriter module.*/  
        void release();
};

/*! The LoadVideo class is designed to manage the interface for loading a video from the local OS. The video can be loaded frame by frame or by simulating a real-time stream. Therefore, in this mode, some frames are discarded if are not required fast enough.
*/
class LoadVideo{
    private:
        string source;
        int warmUp;
        VideoCapture cap;
        int frameProcessed; //count for FPS measure
        
        // variables for realtime video
        bool realtime;
        int webcam;

        // variables to simulate realtime processing on video file
        bool simulateRealtime;
        int millisecPerFrame;
        long long int timeStart;
        int totalFramesGone;

    public:
        /** \brief The constructor method of the class.
         *  \param[in] source The path to the location where the video is saved. If a string of a number is given the class will automatically load the corresponding webcam connected to the device. The count of the webcam starts at 0 and grows.
         *  \param[in] warmUp The warm-up time measured in milliseconds that is used to start the webcam (not used in case of a video).
         *  \param[in] simulateRealtime A flag that, in case of video, says if to simulate a real time processing or not.
         */
        LoadVideo(const string & source="0", int warmUp=1000, bool simulateRealtime=true);
        
        /** \brief Measure the fps of the source video/webcam.
         *  \return The measured fps.
         */
        int sourceFPS();
        
        /** \brief Read a new frame from the input source. If the realtime simulation is running this function will discard useless frames.
         *  \return The read frame.
         */
        Mat read();
        
        /** \brief Calculate the fps rate of the actual reading speed.
         *  \return The calculated fps rate.
         */
        double fps();
        
        /** \brief Measure the milliseconds elapsed since the first execution.
         *  \return The milliseconds elapsed since the first execution.
         */
        long long int elapsed();
};

/** \brief Resize an image to downscale it. The proportions are manteined.
 *  \param[in, out] image The image to be downscaled.
 *  \param[in] downScale The downscale factor.
 */
void downScaleImage(Mat & image, float downScale = 0.6);

/** \brief Resize an image to downscale it. The proportions are manteined.
 *  \param[in, out] image The image to be downscaled.
 *  \param[in] width The width of the image that will be resized.
 */
void downScaleImageTo(Mat & image, int width = 300);

