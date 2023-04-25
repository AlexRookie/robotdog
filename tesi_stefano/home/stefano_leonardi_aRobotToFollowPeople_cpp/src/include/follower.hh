/** 
 *  @file   follower.hh
 *  @brief  The file containing the implementation of the class Follower used as core and connector of all the other classes of the project.
 *  @author Stefano Leonardi
 ***********************************************/
#pragma once

#include "knn.hh"
#include "utils.hh"
#include "videoIO.hh"
#include "dnnModel.hh"

#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using std::cout;
using std::endl;
using std::vector;
using std::unique_ptr;
using std::make_unique;
using cv::Mat;
using cv::Scalar;
using cv::Point2i;
using cv::Rect2d;
using cv::circle;

/*! An enum used to specify the available algorithms of the DetectorModel class. */
enum DetectorNames{SSD=0, YOLO=1};
/*! An enum used to specify the available algorithms of the TrackerNames class. */
enum TrackerNames{CSRT=0, BOOSTING=1, MIL=2, KCF=3, TLD=4, MEDIANFLOW=5, GOTURN=6, MOSSE=7};
/*! An enum used to specify the available algorithms of the ClassifierModel class. */
enum ClassifierNames{RESNET50=0, GOOGLENET=1};

/*! Follower is the main class of the project. It is a fusion of all the other classes and methods in order to effectively track in real-time a person in the real environment. */
class Follower{
    private:
        // Core variables
        unique_ptr<DetectorModel> detector;
        TrackerNames trackerName;
        cv::Ptr<cv::Tracker> tracker;
        unique_ptr<ClassifierModel> classifier;
        unique_ptr<KNN> knn;
        unique_ptr<LoadVideo> videoIn;
        bool useCuda;          
        string negativePeopleDataset;

        // Main flow variables
        Point2i defaultPosition;   //the value returned when the leader is not detected
        Point2i breakPosition;     //the value returned to stop the execution of follow
        bool flagPhase1;           //the controller of pahse 1:slowStart and 2:leaderTracking
        // colors for log visualization
        Scalar colorPhase1Leader;
        Scalar colorPhase1Someone;
        Scalar colorPhase2Leader;
        Scalar colorPhase2LeaderRejected;
        Scalar colorPhase2Someone;
        Scalar colorTracking;
        Scalar colorDrift;
        // variables for the drift optimization
        long long int startDrifting;
        Point2i lastKnownPosition;
        int lastWidth;

        // Hyper-parameters
        int len_slowStartPhase;
        float confidenceThresh;
        int detectionFreq;     
        int k;                 
        float driftRatio;      
        int driftTollerance;   

        // Log-parameters
        bool showLog;          
        unique_ptr<StoreVideo> videoOut;
        int downscaleWidth;    

        // functions
        /** \brief Create a tracker class according to the choice of the user, variable stored internally. */
        void createTracker();
        
        /** \brief Track the leader for the first few seconds.
         *  \details This function rely on the assumption that the leader is always present in the field of view (FOV) of the robot, hence the classification module is not used. In addition this phase is used to add samples to KNN therefore also the tracking is not used because it may be not precise.
         *  \param[in] frame The frame captured from the robot.
         *  \param[out] leaderPosition The computed location of the leader will be stored here. If the location fails a default position is used: (-1, -1). If the user ask to stop the processing a break position is returned: (-2, -2).
        */
        void slowStartPhase(Mat & frame, Point2i & leaderPosition);

        /** \brief Track the leader for the all the period after the first few seconds.
         *  \details This function has no knowledge of the leader behaviour, and no assumptions can be done. Therefore the detection and tracking phases are continuosly alternated to follow it, while KNN is trained to recognise it always better.
         *  \param[in] frame The frame captured from the robot.
         *  \param[out] leaderPosition The computed location of the leader will be stored here. If the location fails a default position is used: (-1, -1). If the user ask to stop the processing a break position is returned: (-2, -2).
        */
        void leaderTrackingPhase(Mat & frame, Point2i & leaderPosition);
        

        /** \brief Internally load an image used as negative sample in the training, and send it to KNN to improve its feature space. */
        void addNegativeToKNN();

        /** \brief Add multiple detections to the KNN feature space, according to their labels.
         *  \param[in] frame The frame on which the detections were performed.
         *  \param[in] detections The output of the feed function of a DetectorModel class. AKA multiple BBs, with labels and confidences.
         *  \param[in] leaderIndex Only one leader can exist, this represent the index of its detection. A negative index, as -1, is considered as: no leader exist.
        */
        void addBBsToKNN(const Mat & frame, const vector<vector<int> > & detections, int leaderIndex=-1);

        /** \brief Draw multiple detections on the given frame.
         *  \param[in, out] frame The frame on which the detections will be drawn.
         *  \param[in] detections The output of the feed function of a DetectorModel class. AKA multiple BBs, with labels and confidences.
         *  \param[in] colorSomeone The color used for a normal detection.
         *  \param[in] leaderIndex Only one leader can exist, this represent the index of its detection. A negative index, as -1, is considered as: no leader exist.
         *  \param[in] colorLeader The color used for a normal detection.
         *  \param[in] drawCenter A flag, if true the center of the BBs is highlighted with a small concentric circle, otherwise only the BB appears.
        */
        void drawDetectionsToFrame(Mat & frame, const vector<vector<int> > & detections, const Scalar & colorSomeone, int leaderIndex=-1, const Scalar & colorLeader = Scalar(0,0,0), bool drawCenter=true);

        // utility
        /** \brief Load an image used as negative sample in the training procedure of KNN.
         *  \details The image is loaded from a dataset of 5180 images used as negative samples. Each image is the cropped BB of a person.
         *  \param[out] The image loaded is returned.
        */
        void loadNegativePerson(Mat & image);
        
        /** \brief Calculate the centre of the bounding box, defined with the four dimension given.
         *  \param[in] left The left value representing the BB.
         *  \param[in] top The top value representing the BB.
         *  \param[in] width The width value representing the BB.
         *  \param[in] height The height value representing the BB.
        */
        Point2i centerBB(int left, int top, int width, int height);
        
        /** \brief Calculate the centre of the bounding box, defined by the rectangle of double given.
         *  \param[in] The rectangle of which the centre should be computed.
        */
        Point2i centerBB(const Rect2d & rect);
        
        /** \brief Add the FPS measure to the frame and then show it with the StoreVideo class.
         *  \param[in] The frame to be shown.
        */
        bool showFrame(Mat & frame);
    

    public:
        /** \brief The constructor method that create and initialize all the required classes.
         *  \param[in] detectorName The enum (DetectorNames) that identify which algorithm should be used to perform the detection task.
         *  \param[in] trackerName The enum (TrackerNames) that identify which algorithm should be used to perform the tracking task.
         *  \param[in] classifierName The enum (ClassifierNames) that identify which algorithm should be used to perform the classification task.
         *  \param[in] sourceVideo The path to the source video. If a string of a number is given the corresponding webcam connected to the device is automatically picked. The count of the webcam starts at 0 and grows.
         *  \param[in] simulateRealtime A flag, if true the source video is processed in order to simultate a realtime processing, otherwise the frames are processed one after the other.
         *  \param[in] useCuda A flag, if true abilitate the processing of openCV with a GPU on cuda, otherwise the CPU is used.
         *  \param[in] negativePeopleDataset The path to the location of the dataset of people used as negative samples.
        */
        Follower(\
            DetectorNames detectorName = SSD,\
            TrackerNames trackerName = KCF,\
            ClassifierNames classifierName = RESNET50,\
            string sourceVideo = "0",\
            bool simulateRealtime = true,\
            bool useCuda = false,
            string negativePeopleDataset = "./imagesIn/negativePeople_dataset/repeatedPeople/");

        /** \brief An extension of the constructor.
         *  \details It is used to set hyper parameters neccessary to tune the algorithms behaviours.
         *  \param[in] len_slowStartPhase The lenght, in milliseconds, of the first phase (AKA slow start phase).
         *  \param[in] confidenceThresh The threshold on the confidence used choose if a detection should be kept or not.
         *  \param[in] detectionFreq It guarantee that a detection is performed once every given milliseconds.
         *  \param[in] k The k value used for KNN. 
         *  \param[in] driftRatio A multiply factor that influence the growth of the of the drift tolerance radius.
         *  \param[in] driftTollerance An additive factor that set the minimum value of the drift tolerance radius.
        */
        void setHyperParam(\
            int len_slowStartPhase = 5000,\
            float confidenceThresh = 0.4,\
            int detectionFreq = 800,\
            int k = 10,\
            float driftRatio = 0.05,\
            int driftTollerance = 30);

        /** \brief An extension of the constructor.
         *  \details It is used to set the parameters that tune the behaviour of the log.
         *  \param[in] showLog A flag, if true the generated video is saved to the OS and it may also be shown while it is generated.
         *  \param[in] destVideo The path where to save the generated video.
         *  \param[in] downscaleWidth To reduce the size of the saved video each frame is resized. This value specify the width of the resize.
         *  \param[in] destFps The frequency for the frames in the video that should be saved. 
        */
        void setLogParam(\
            bool showLog = false,\
            string destVideo = "out",\
            int downscaleWidth = 600,\
            int destFps = 10);

        /** \brief The main function of this class.
         *  \details Internally follow the leader with a combination of three methods: detection, tracking and classification.
         *  \return The computed location of the leader will be stored here. If the location fails a default position is used: (-1, -1). If the user ask to stop the processing, or the video source is end, a break position is returned: (-2, -2).
         */
        Point2i follow();
};