/** 
 *  @file   main3D.cc
 *  @brief  A wrapper file used for run the functionalities of the Follower3D class.
 *  @author Paolo Bevilacqua & Stefano Leonardi
 ***********************************************/

#include "follower3D.hh"
#include "logger3D.hh"
#include "zmq/Publisher.hh"
#include "zmq/Subscriber.hh"
#include "json.hpp"

#include <unistd.h>
#include <signal.h>

#include <opencv2/highgui.hpp>  //command line parser
using std::string;
using nlohmann::json;

#define OFFSET_CAMERA_ROBOT 0.28 //0.1 // [m]

// usage:
// cls && ./compile.sh
// ./build/buildLinux/main -h
// ./build/buildLinux/main -d=0 -t=3 -c=1 -i="./videosIn/shelfy_dataset/v04-simpleIntersection.mp4" -l=1 -o="videosOut/intersection"
// ./build/buildLinux/main -d=1 -t=0 -c=0 -i="./videosIn/shelfy_dataset/v04-simpleIntersection.mp4" -l=1 -o="videosOut/intersection" --detectionFreq=1500

/*! The list of all the accepted command line parameters. */
const char* keys =
"{_00 help h        |<none>| Usage examples: \n\t\t./build/buildLinux/main -d=0 -t=3 -c=1 -i=\"./videosIn/shelfy_dataset/v04-simpleIntersection.mp4\" -l=1 -o=\"videosOut/intersection\"\n}"
// Follower costructor
"{_01 detector      d           |0|     int: SSD=0, YOLO=1 -> The enum (DetectorNames) that identify which algorithm should be used to perform the detection task.\n}"
"{_02 tracker       t           |3|     int: CSRT=0, BOOSTING=1, MIL=2, KCF=3, TLD=4, MEDIANFLOW=5, GOTURN=6, MOSSE=7 -> The enum (TrackerNames) that identify which algorithm should be used to perform the tracking task.\n}"
"{_03 classifier    c           |3|     int: RESNET50=0, GOOGLENET=1, GOOGLENET_TRT=2, OSNET=3 -> The enum (ClassifierNames) that identify which algorithm should be used to perform the classification task.\n}"
"{_04 sourceVideo   i           |0|     string: The path to the source video. If a string of a number is given the corresponding webcam connected to the device is automatically picked. The count of the webcam starts at 0 and grows.\n}"
"{_05 simulateRealtime          |true|  bool: A flag, if true the source video is processed in order to simultate a realtime processing, otherwise the frames are processed one after the other.\n}"
"{_06 useCuda                   |false| bool: A flag, if true abilitate the processing of openCV with a GPU on cuda, otherwise the CPU is used.\n}"
"{_07 negativePeopleDataset     |./imagesIn/negativePeople_dataset/repeatedPeople/| string: The path to the location of the dataset of people used as negative samples.\n}"

// set Hyper parameter 
"{_08 len_slowStartPhase        |10000| int: The lenght, in milliseconds, of the first phase (AKA slow start phase).\n}"
"{_09 confidenceThresh          |0.4|   float: The threshold on the confidence used choose if a detection should be kept or not.\n}"
"{_10 detectionFreq             |800|   int: It guarantee that a detection is performed once every given milliseconds.\n}"
"{_11 k                         |10|    int: The k value used for KNN. \n}"
"{_12 driftRatio                |0.05|  float: A multiply factor that influence the growth of the of the drift tolerance radius.\n}"
"{_13 driftTollerance           |30|    int: An additive factor that set the minimum value of the drift tolerance radius.\n}"

// set Log parameter
"{_14 showLog     l             |true|  bool: A flag, if true the generated video is saved to the OS and it may also be shown while it is generated.\n}"
"{_15 destVideo   o             |out|   string: The path where to save the generated video.\n}"
"{_16 downscaleWidth            |600|   int: To reduce the size of the saved video each frame is resized. This value specify the width of the resize.\n}"
"{_17 destFps                   |10|    int: The frequency for the frames in the video that should be saved.\n}"

"{_18 pubAddress                |tcp://*:3210|  string: IP address of ZMQ publisher.\n}"
"{_19 subAddress                |tcp://127.0.0.1:3211|  string: (ZMQ) IP address to subscribe to.\n}"
"{_20 recordMode                |false|  bool: Enable recording mode.\n}"
;


std::string subAddressHuman = "tcp://127.0.0.1:3218"; //parser.get<string>("pubAddress");
std::string subAddressLoc   = "tcp://10.196.80.135:9207"; 
std::string subAddressOdom  = "tcp://10.196.80.135:9813"; 

std::string subTopicHuman = "HUM_POS"; 
std::string subTopicLoc   = "POS";
std::string subTopicOdom  = "ODOM"; 


volatile bool terminating = false;
void ctrlc_callback(int) {
    terminating = true;
}



void recordingMode(cv::CommandLineParser const & parser);

/** \brief  The main function used as a wrapper for the Follower class.
 *  \param[in] argc The number of command line parameters. 
 *  \param[in] argv The list of string passed from the command line.
 *  \return None
*/
int main(int argc, char** argv){
    signal(SIGINT, ctrlc_callback);

    // Parse the arguments
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("This is a sample main code to run the Follower class and parse all its arguments.");
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    if (parser.get<bool>("recordMode")) {
        recordingMode(parser);
    }
    else {

        // Initialize the Follower class
        Follower3D follower = Follower3D((DetectorNames)parser.get<int>("detector"), (TrackerNames)parser.get<int>("tracker"), (ClassifierNames)parser.get<int>("classifier"), parser.get<bool>("useCuda"), parser.get<string>("negativePeopleDataset"));
        
        follower.setHyperParam(parser.get<int>("len_slowStartPhase"), parser.get<float>("confidenceThresh"), parser.get<int>("detectionFreq"), parser.get<int>("k"), parser.get<float>("driftRatio"), parser.get<int>("driftTollerance"));
        
        follower.setLogParam(parser.get<bool>("showLog"), parser.get<string>("destVideo"), parser.get<int>("downscaleWidth"), parser.get<int>("destFps"));

        string pubAddress = parser.get<string>("pubAddress");
        Publisher pub(pubAddress);

        ////////////////////////////////////////////////////////////////////////////
        // SUBSCRIBER TO HUMAN POSE
        ////////////////////////////////////////////////////////////////////////////
        
        struct HumPose {
            double x = 0, y = 0;
            bool valid = false;
            long long int timestamp = 0;
            std::mutex mtx;
        };

        Subscriber sub2D;
        HumPose trackedPose;
        sub2D.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
            try {
                json jobj = json::parse(std::string(buf, size));
                long long int timestamp = jobj.at("ts");
                bool valid = jobj.at("valid");
                double x = jobj.at("x");
                double y = jobj.at("y");
                {
                    std::unique_lock<std::mutex> lck(trackedPose.mtx);
                    trackedPose.x = x;
                    trackedPose.y = y;
                    trackedPose.valid = valid;
                    trackedPose.timestamp = timestamp;
                }
            }
            catch(std::exception &e) {
                std::cerr << "Error parsing human tracking data: " << e.what() << std::endl;
            }
        });
        sub2D.start(subAddressHuman, subTopicHuman);

        
        ////////////////////////////////////////////////////////////////////////////
        // SUBSCRIBER TO ODOM
        ////////////////////////////////////////////////////////////////////////////

        struct Odometry {
            double x, y, theta;
            double v, omega;
            long long int ts;
            std::mutex mtx;
        };

        Subscriber subOdom;
        Odometry odomData;
        subOdom.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
            nlohmann::json j;

            try {
                j = nlohmann::json::parse(std::string(buf, size));   
                double x, y, theta;
                double v, omega;
                long long int ts;
                ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                
                x = j["x"];
                y = j["y"];
                theta = j["theta"];
                v = j["v"];
                omega = j["omega"];
                
                {
                    std::unique_lock<std::mutex> lock(odomData.mtx);
                    odomData.x = x;
                    odomData.y = y;
                    odomData.theta = theta;
                    odomData.v = v;
                    odomData.omega = omega;
                    odomData.ts = ts;
                }

            }
            catch(std::exception &e) {
                std::cerr << "Error parsing odom data: " << e.what() << std::endl;
            }
        });
        subOdom.start(subAddressOdom, subTopicOdom);



        ////////////////////////////////////////////////////////////////////////////
        // SUBSCRIBER TO LOC
        ////////////////////////////////////////////////////////////////////////////

        struct Pose {
            double x, y, theta;
            long long int ts;
            std::mutex mtx;
        };

        Subscriber subLoc;
        Pose locData;
        subLoc.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
            nlohmann::json j;
            
            try {
                j = nlohmann::json::parse(std::string(buf, size));
                double x, y, theta;
                long long int ts;
                x = j.at("loc_data").at("x");
                y = j.at("loc_data").at("y");
                theta = j.at("loc_data").at("theta");
                ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                {
                    std::unique_lock<std::mutex> lock(locData.mtx);
                    locData.x = x;
                    locData.y = y;
                    locData.theta = theta;
                    locData.ts = ts;
                }
            }
            catch(std::exception &e)  {
                std::cerr << "\"" << std::string(buf, size) << "\"" << std::endl;
                std::cerr << "error parsing loc data: " << e.what() << std::endl;
            }
        });
        subLoc.start(subAddressLoc, subTopicLoc);




        
        
        // process frame by frame
        TimedPose position;
        
        for(; !terminating; ){ 
            TimedPose pose2Drel;
            {
                std::unique_lock<std::mutex> lck(trackedPose.mtx);
                pose2Drel.x = trackedPose.x;
                pose2Drel.y = trackedPose.y;
                pose2Drel.valid = trackedPose.valid;
                pose2Drel.timestamp = trackedPose.timestamp;
            }

            bool est_pose = false;
            double x_h_cam = 0;
            double y_h_cam = 0;
            if (pose2Drel.valid) {
                est_pose = true;
                // ROTO-TRANSLATE and convert from ROBOT to CAMERA
                pose2Drel.x -= OFFSET_CAMERA_ROBOT;
                x_h_cam = -pose2Drel.y;
                y_h_cam = pose2Drel.x;
            }
            
            position = follower.follow(est_pose, x_h_cam, y_h_cam);
            
            //if (!position.valid) continue;
            
            // roto-translate according to camera POSE wrt ROBOT
            double x = position.y;
            double y = -position.x;
            x += OFFSET_CAMERA_ROBOT;
            
            json jobj;
            jobj["ts"] = position.timestamp;
            jobj["valid"] = position.valid;
            jobj["x"] = x;
            jobj["y"] = y;
            
            std::string jmsg = jobj.dump();
            pub.send("T3D", jmsg.c_str(), jmsg.size());
        }
    }

    std::cout << "\nTHE END\n\n";
    return(0);
}



void recordingMode(cv::CommandLineParser const & parser) {
    Logger3D logger = Logger3D((DetectorNames)parser.get<int>("detector"), parser.get<string>("destVideo"), parser.get<float>("confidenceThresh"));
        
    while(!terminating){ 
        logger.log();       
    }
}
