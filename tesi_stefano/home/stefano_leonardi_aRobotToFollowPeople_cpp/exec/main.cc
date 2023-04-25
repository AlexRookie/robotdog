/** 
 *  @file   main.cc
 *  @brief  A wrapper file used for run the functionalities of the Follower class.
 *  @author Stefano Leonardi
 ***********************************************/

#ifdef REALSENSE_3D
#include "follower3D.hh"
#else
#include "follower.hh"
#endif 

#include <opencv2/highgui.hpp>  //command line parser
using std::string;

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
"{_03 classifier    c           |0|     int: RESNET50=0, GOOGLENET=1 -> The enum (ClassifierNames) that identify which algorithm should be used to perform the classification task.\n}"
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
;

/** \brief  The main function used as a wrapper for the Follower class.
 *  \param[in] argc The number of command line parameters. 
 *  \param[in] argv The list of string passed from the command line.
 *  \return None
*/
int main(int argc, char** argv){

    // Parse the arguments
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("This is a sample main code to run the Follower class and parse all its arguments.");
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    #ifdef REALSENSE_3D
    // Initialize the Follower class
    Follower3D follower = Follower3D((DetectorNames)parser.get<int>("detector"), (TrackerNames)parser.get<int>("tracker"), (ClassifierNames)parser.get<int>("classifier"), parser.get<bool>("useCuda"), parser.get<string>("negativePeopleDataset"));
    #else
    // Initialize the Follower class
    Follower follower = Follower((DetectorNames)parser.get<int>("detector"), (TrackerNames)parser.get<int>("tracker"), (ClassifierNames)parser.get<int>("classifier"), parser.get<string>("sourceVideo"), parser.get<bool>("simulateRealtime"), parser.get<bool>("useCuda"), parser.get<string>("negativePeopleDataset"));
    #endif
    

    follower.setHyperParam(parser.get<int>("len_slowStartPhase"), parser.get<float>("confidenceThresh"), parser.get<int>("detectionFreq"), parser.get<int>("k"), parser.get<float>("driftRatio"), parser.get<int>("driftTollerance"));

    follower.setLogParam(parser.get<bool>("showLog"), parser.get<string>("destVideo"), parser.get<int>("downscaleWidth"), parser.get<int>("destFps"));

    // process frame by frame
    #ifdef REALSENSE_3D
    TimedPose position;
    #else
    Point2i position;
    #endif
    // the point (-2, -2) is used to stop the tracking procedure
    while(position.x != -2){ 
        position = follower.follow();
        cout << "Actual position is: " << position << endl;
    }

    std::cout << "\nTHE END\n\n";
return(0);
}


