/** 
 *  @file   dnnModel.hh
 *  @brief  A file containing the implementation of the dnn based classes ClassifierModel and DetectorModel, and the realtive sub-classes.
 *  @author Stefano Leonardi
 ***********************************************/
#pragma once
#include "utils.hh"

#ifndef NO_CUDA
    #include "trtNet.h"
#endif

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <fstream>

using std::vector;
using std::string;
using std::ifstream;
using std::cout;
using std::endl;
using cv::Mat;
using cv::Size;
using cv::Rect;
using cv::Point;
using cv::Scalar;
using cv::dnn::Net;
using cv::dnn::NMSBoxes;
using cv::dnn::blobFromImage;
using cv::dnn::readNetFromONNX;
using cv::dnn::readNetFromCaffe;
using cv::dnn::readNetFromDarknet;

/*! The DnnModel is an abstract class that implement only the main functions common for every DNN based class.*/
class DnnModel{
    protected:
        /*! Path to the model.*/
        string modelPath1;	
        /*! Path to the second part of the model.*/
        string modelPath2;	
        /*! The opencv network model.*/
        Net net;	        
        /*! The name of the layer where to end the DNN.*/
        vector<string> layerNames;	

        // utility
        /** \brief Print to the console the dimensions of the given Mat.
         *  \param[in] m The Mat to be shown.
         *  \param[in] name The name to be printed.
        */
        void printMatDims(const Mat & m, const string name);

        /** \brief Perform a range operation on one dimension of the matrix.
         *  \param[in] m The Mat to be scan.
         *  \param[out] enc The extracted portion of the Mat.
         *  \param[in] dim The dimension on which the scan is performed.
         *  \param[in] start The start index of the range.
         *  \param[in] end The end index of the range (not included).
        */
        void range1D(const Mat & m, vector<float> & enc, int dim, int start, int end);

    public:
        /** \brief The constructor method of the class. Each value is simply stored internally.
         *  \param[in] modelPath1 The path to the first part of model file contining the implementation of the DNN.
         *  \param[in] modelPath2 The path to the second part of model file contining the implementation of the DNN.
         *  \param[in] layerNames Represent the list of the names of the layers where the output of the DNN is generated.
         *  \param[in] gpuBackend A flag, if set to true openCV will use a GPU otherwise not.
        */
        DnnModel(string modelPath1, string modelPath2, const vector<string> & layerNames, bool gpuBackend);
        
        /** \brief Abilitate the processing of openCV with a GPU on cuda. */
        void useCuda();
        

        /** \brief An image is trasformated into a blob in order to feed a NN.
         *  \param[in] image The image to be elaborated.
         *  \param[out] blob The generated blob.
        */
        virtual void blob(const Mat & image, Mat & blob) = 0;

        /** \brief Feed the NN with a blob.
         *  \param[in] blob The blob to be fed into the NN.
        */
        void setInput(const Mat & blob);
        
        /** \brief Perform a forward pass to the NN, to generate its output.
         *  \param[out] detections The output of the NN.
        */
        void forward(vector<Mat> & detections);
};

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////   CLASSIFIER   ////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
/*! The ClassifierModel is an abstract class that specify the requirements and implement detailed functions for the object classifier dnn models.*/
class ClassifierModel : public DnnModel {
    public:
        /** \brief The constructor method of the class. Each value is simply stored internally.
         *  \param[in] modelPath1 The path to the first part of model file contining the implementation of the DNN.
         *  \param[in] modelPath2 The path to the second part of model file contining the implementation of the DNN.
         *  \param[in] layerNames Represent the list of the names of the layers where the output of the DNN is generated.
         *  \param[in] gpuBackend A flag, if set to true openCV will use a GPU otherwise not.
        */
        ClassifierModel(string modelPath1, string modelPath2, const vector<string> & layerNames, bool gpuBackend);


        /** \brief The feed operation allow to execute the default serie of operations. 
         *  \details An image is taken, pre-processed, elaborated with the NN, and then the output is refined before returning it. I suggest to always use this function. The output is a vector of n float values (in the range [0, 1]). This values represent the encoding of the input image.
         *  \param[in] image The image to be elaborated.
         *  \param[out] output The refined output of the NN.
        */
        virtual void feed(const Mat & image, vector<float> & output);

    private:
        /** \brief An image is trasformated into a blob in order to feed a NN.
         *  \param[in] image The image to be elaborated.
         *  \param[out] blob The generated blob.
        */
        virtual void blob(const Mat & image, Mat & blob);

        /** \brief The output of the NN is elaborated to retrieve only the useful information in a condensed way.
         *  \details The output is a vector of n float values (in the range [0, 1]). This values represent the encoding of the input image.
         *  \param[in] detections The output of the NN that need to be refined.
         *  \param[out] output The refined output of the NN.
        */
        void processDnnOutput(const vector<Mat> & detections, vector<float> & output);
};


/*! The ResNet50  class is a dnn object classifier, that generate an encoding of an image in 2048 dimensions.*/
class ResNet50 : public ClassifierModel{
    public:
        /** \brief The constructor method of the class, the default are set according to the standard structure of the project.
         *  \param[in] modelPath1 The path to the first part of model file contining the implementation of the DNN.
         *  \param[in] layerNames Represent the list of the names of the layers where the output of the DNN is generated.
         *  \param[in] gpuBackend A flag, if set to true openCV will use a GPU otherwise not.
        */
        ResNet50(\
            bool gpuBackend = false, \
            string modelPath1 = "./models/resnet50-caffe2/resnet50-caffe2.onnx", \
            const vector<string> & layerNames = vector<string>{"OC2_DUMMY_0"});


};


/*! The GoogleNet class is a dnn object classifier, that generate an encoding of an image in 1024 dimensions.*/
class GoogleNet : public ClassifierModel{
    public:
        /** \brief The constructor method of the class, the default are set according to the standard structure of the project.
         *  \param[in] modelPath1 The path to the first part of model file contining the implementation of the DNN.
         *  \param[in] modelPath2 The path to the second part of model file contining the implementation of the DNN.
         *  \param[in] layerNames Represent the list of the names of the layers where the output of the DNN is generated.
         *  \param[in] gpuBackend A flag, if set to true openCV will use a GPU otherwise not.
        */
        GoogleNet(\
            bool gpuBackend = false, \
            string modelPath1 = "./models/googleNet/bvlc_googlenet.prototxt", \
            string modelPath2 = "./models/googleNet/bvlc_googlenet.caffemodel", \
            const vector<string> & layerNames = vector<string>{"pool5/7x7_s1"});
};


class GoogleNetTRT : public ClassifierModel {
    public:
        
        GoogleNetTRT();

        virtual void feed(const cv::Mat & image, std::vector<float> & output) override;

    private:
        
        void init();
        
        static cv::Mat normalizeImg(cv::Mat const & img);
        static void hwcToChw(cv::Mat const & src, std::vector<float> & data);

#ifndef NO_CUDA
        trtnet::TrtGooglenet net; 
#endif

        static const std::string DEPLOY_ENGINE;
        static const cv::Scalar PIXEL_MEANS;
        static const int ENGINE_SHAPE0[3];
        static const int ENGINE_SHAPE1[3];
        static const cv::Size RESIZED_SHAPE;
        static const bool doCropping;
};


class OsNet : public ClassifierModel{
    public:
        /** \brief The constructor method of the class, the default are set according to the standard structure of the project.
         *  \param[in] modelPath1 The path to the first part of model file contining the implementation of the DNN.
         *  \param[in] layerNames Represent the list of the names of the layers where the output of the DNN is generated.
         *  \param[in] gpuBackend A flag, if set to true openCV will use a GPU otherwise not.
        */
        OsNet(\
            bool gpuBackend = false, \
            string modelPath1 = "./models/osNet/osnet_x1_0.onnx", \
            const vector<string> & layerNames = vector<string>{"output"});

    protected:
        virtual void blob(const Mat & image, Mat & blob) override;
};



///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////   DETECTOR   //////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
/*! The order of the elements returned from the feed operation. */
enum DetectionElements{CLASSID=0, CONFIDENCE=1, LEFT=2, TOP=3, WIDTH=4, HEIGHT=5};

/*! The DetectorModel is an abstract class that specify the requirements and implement detailed functions for the object detector dnn models.*/
class DetectorModel : public DnnModel{
    protected:
        /*! Path to the file where the names of detection are saved. */
        string namesPath;		
        /*! The list of names of detection. */
        vector<string> classes;	
        /*! Optimization that store the ID of the class "person". */
        int idClassPerson;      

    public:
        /** \brief The constructor method of the class. Each value is simply stored internally.
         *  \param[in] modelPath1 The path to the first part of model file contining the implementation of the DNN.
         *  \param[in] modelPath2 The path to the second part of model file contining the implementation of the DNN.
         *  \param[in] namesPath  The path to the file where the names of object detection are saved. Each line is a new name.
         *  \param[in] layerNames Represent the list of the names of the layers where the output of the DNN is generated.
         *  \param[in] gpuBackend A flag, if set to true openCV will use a GPU otherwise not.
        */
        DetectorModel(string modelPath1, string modelPath2, string namesPath, const vector<string> & layerNames, bool gpuBackend);
        

        /** \brief The feed operation allow to execute the default serie of operations. 
         *  \details An image is taken, pre-processed, elaborated with the NN, and then the output is refined before returning it. I suggest to always use this function. The output is a vector of n vectors each one representing a detection. Each detection is a vector of 6 int that respectively represent: {0 = the id of the class found, 1 = the confidence of the match, 2-3 the left-top point and 4-5 the width-height of the bounding box (BB) delimiting the detection found}.
         *  \param[in] image The image to be elaborated.
         *  \param[out] output The refined output of the NN.
         *  \param[in] confidenceThresh The lower-bound threshold to choose if a detection should be accepted or rejected.
        */
        void feed(const Mat & image, vector<vector<int> > & output, float confidenceThresh=0.5);

        /** \brief A wrapper method for the feed function. It guarantees that the detections retrieved belongs only to the class "person".
         *  \details The output is in the same format of the feed function result.
         *  \param[in] image The image to be elaborated.
         *  \param[out] output The refined output of the NN.
         *  \param[in] confidenceThresh The lower-bound threshold to choose if a detection should be accepted or rejected.
        */
        void detectPeopleOnly(const Mat & image, vector<vector<int> > & output, float confidenceThresh=0.5);

        /** \brief The output of the NN is elaborated to retrieve only the useful information in a condensed way.
         *  \details The output is in the same format of the feed function result.
         *  \param[in] detections The output of the NN that need to be refined.
         *  \param[out] output The refined output of the NN.
         *  \param[in] size The original dimensions of the input image, that are used to retrive the coordinate as absolute values and not as percentage.
         *  \param[in] confidenceThresh The lower-bound threshold to choose if a detection should be accepted or rejected.
        */
        virtual void processDnnOutput(const vector<Mat> & detections, vector<vector<int> > & output, const Size & size, float confidenceThresh=0.5) = 0;


        /** \brief Draw a single detection to a frame. The draw include the bounding box, the class and the confidence.
         *  \param[in] detection A detection coming form the refined output of the NN.
         *  \param[in, out] frame the frame on which the function will draw the predictions.
         *  \param[in] color The color used to draw.
         *  \param[in] showConfidence A flag, if true the confidence value will be shown, otherwise not.
        */
        void drawOnePrediction(const vector<int> & detection, Mat & frame, const Scalar & color = {0, 0, 0}, bool showConfidence=true);
 
        /** \brief Draw multiple detections to a frame. The draw include the bounding box, the class and the confidence.
         *  \param[in] detections The refined output of the NN.
         *  \param[in, out] frame the frame on which the function will draw the predictions.
         *  \param[in] color The color used to draw.
         *  \param[in] showConfidence A flag, if true the confidence value will be shown, otherwise not.
        */
        void drawPredictions(const vector<vector<int> > & detections, Mat & frame, const Scalar & color = {0, 0, 0}, bool showConfidence=true);
};


/*! The MobileNetSSD class is a dnn object detector, that localize with bounding boxes a lot of different classes of elements.*/
class MobileNetSSD : public DetectorModel{
    public:
        /** \brief The constructor method of the class, the default are set according to the standard structure of the project.
         *  \param[in] modelPath1 The path to the first part of model file contining the implementation of the DNN.
         *  \param[in] modelPath2 The path to the second part of model file contining the implementation of the DNN.
         *  \param[in] namesPath  The path to the file where the names of object detection are saved. Each line is a new name.
         *  \param[in] layerNames Represent the list of the names of the layers where the output of the DNN is generated.
         *  \param[in] gpuBackend A flag, if set to true openCV will use a GPU otherwise not.
        */
        MobileNetSSD(\
            string modelPath1 = "./models/mobileNet_SSD/MobileNetSSD_deploy.prototxt", \
            string modelPath2 = "./models/mobileNet_SSD/MobileNetSSD_deploy.caffemodel", \
            string namesPath  = "./models/mobileNet_SSD/SSDnames.txt", \
            const vector<string> & layerNames = vector<string>{"detection_out"}, \
            bool gpuBackend = false);


        /** \brief An image is trasformated into a blob in order to feed a NN.
         *  \param[in] image The image to be elaborated.
         *  \param[out] blob The generated blob.
        */
        void blob(const Mat & image, Mat & blob);

        /** \brief The output of the NN is elaborated to retrieve only the useful information in a condensed way.
         *  \details The output is in the same format of the feed function result.
         *  \param[in] detections The output of the NN that need to be refined.
         *  \param[out] output The refined output of the NN.
         *  \param[in] size The original dimensions of the input image, that are used to retrive the coordinate as absolute values and not as percentage.
         *  \param[in] confidenceThresh The lower-bound threshold to choose if a detection should be accepted or rejected.
        */
        void processDnnOutput(const vector<Mat> & detections, vector<vector<int> > & output, const Size & size, float confidenceThresh=0.5);
};


/*! The YOLOv3 class is a dnn object detector, that localize with bounding boxes a lot of different classes of elements.*/
class YOLOv3 : public DetectorModel{
    private:
        float confidenceNMS;

    public:
        /** \brief The constructor method of the class, the default are set according to the standard structure of the project.
         *  \param[in] modelPath1 The path to the first part of model file contining the implementation of the DNN.
         *  \param[in] modelPath2 The path to the second part of model file contining the implementation of the DNN.
         *  \param[in] namesPath  The path to the file where the names of object detection are saved. Each line is a new name.
         *  \param[in] layerNames Represent the list of the names of the layers where the output of the DNN is generated.
         *  \param[in] gpuBackend A flag, if set to true openCV will use a GPU otherwise not.
         *  \param[in] confidenceNMS The lower-bound threshold to the Non-Maxima Suppression algorithm that will refine the detections of YOLO that are too close to each other.
        */
       YOLOv3(\
            string modelPath1 = "./models/yoloV3-coco/yolov3.cfg", \
            string modelPath2 = "./models/yoloV3-coco/yolov3.weights", \
            string namesPath  = "./models/yoloV3-coco/coco.names", \
            const vector<string> & layerNames = vector<string>{"yolo_82", "yolo_94", "yolo_106"}, \
            bool gpuBackend = false,
            float confidenceNMS = 0.3);


        /** \brief An image is trasformated into a blob in order to feed a NN.
         *  \param[in] image The image to be elaborated.
         *  \param[out] blob The generated blob.
        */
        void blob(const Mat & image, Mat & blob);

        /** \brief The output of the NN is elaborated to retrieve only the useful information in a condensed way.
         *  \details The output is in the same format of the feed function result.
         *  \param[in] detections The output of the NN that need to be refined.
         *  \param[out] output The refined output of the NN.
         *  \param[in] size The original dimensions of the input image, that are used to retrive the coordinate as absolute values and not as percentage.
         *  \param[in] confidenceThresh The lower-bound threshold to choose if a detection should be accepted or rejected.
        */
        void processDnnOutput(const vector<Mat> & detections, vector<vector<int> > & output, const Size & size, float confidenceThresh=0.5);
};



