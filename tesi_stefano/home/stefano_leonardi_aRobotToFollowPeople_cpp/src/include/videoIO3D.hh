/** 
 *  @file   videoIO3D.hh
 *  @brief  A file containing the implementation of the LoadVideo3D class.
 *  @author Paolo Bevilacqua
 ***********************************************/
#pragma once //force the compiler to load this file only once (the same of #define UNIQUE_NAME)

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "utils.hh"
#include "single_thread.hh"

using std::cout;
using std::endl;
using std::string;
using std::unique_ptr;
using std::mutex;

using cv::Mat;
using cv::waitKey;

using rs2::context;
using rs2::pipeline;
using rs2::config;
using rs2::device;

struct FrameData3D {
    Mat frame;
    std::unique_ptr<rs2::depth_frame> depth_frame;

    long long int lastTime = -1;
    // TODO: ...

    FrameData3D() {}
    FrameData3D(FrameData3D const & other)
    {
        clone(other);
    }
    FrameData3D& operator=(FrameData3D const & other)
    {
        clone(other);        
        return *this;
    }

    void clone(FrameData3D const & other) {
        frame = other.frame.clone();

        if (other.depth_frame)
            depth_frame = std::make_unique<rs2::depth_frame>(*other.depth_frame);
        lastTime = other.lastTime;
    }
};

class LoadVideo3D {
    private:
        int warmUp;
        int frameProcessed; //count for FPS measure

        // variables to simulate realtime processing on video file
        long long int timeStart;

        // 3D RS data
        unique_ptr<context> ctx;
        unique_ptr<pipeline> pipe;
        unique_ptr<config> cfg;

        unique_ptr<rs2::stream_profile> rgb_stream, depth_stream;

        unique_ptr<rs2_intrinsics> rgb_intrinsics, depth_intrinsics;

        unique_ptr<rs2::align> aligner;

        unique_ptr<single_thread> bg_thread;

        mutex dataMtx;
        std::condition_variable dataCV;
        FrameData3D data;

        void init3Dcamera();
        
    public:
        /** \brief The constructor method of the class.
         *  \param[in] warmUp The warm-up time measured in milliseconds that is used to start the webcam (not used in case of a video).
         */
        LoadVideo3D(int warmUp=1000);

        ~LoadVideo3D();
        
        /** \brief Measure the fps of the source video/webcam.
         *  \return The measured fps.
         */
        int sourceFPS();
        
        /** \brief Read a new frame from the input source. If the realtime simulation is running this function will discard useless frames.
         *  \return The read frame.
         */
        FrameData3D read();
        
        /** \brief Calculate the fps rate of the actual reading speed.
         *  \return The calculated fps rate.
         */
        double fps();
        
        /** \brief Measure the milliseconds elapsed since the first execution.
         *  \return The milliseconds elapsed since the first execution.
         */
        long long int elapsed();

        bool worldCoordinates(FrameData3D const & fd, int i, int j, float& x, float& y, float& z);
};
