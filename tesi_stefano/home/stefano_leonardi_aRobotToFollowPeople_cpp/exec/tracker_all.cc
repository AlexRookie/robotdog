/** @file   tracker_all.cc
 *  @brief  A working example of many different tracking algorithms.
 *  @author Stefano Leonardi
 ***********************************************/
// The original code manage multiple bounding boxes and multiple tracker simultaneously.
// Check out that one if needed.

// documentation tracker:       https://docs.opencv.org/3.4/d0/d0a/classcv_1_1Tracker.html
// documentation multi-tracker: https://docs.opencv.org/3.4/d8/d77/classcv_1_1MultiTracker.html

// usage: cls && ./compile.sh && ./build/buildLinux/trackerAll

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <exception>

#include "videoIO.hh"
#include "utils.hh"

using namespace cv;
using namespace std;

// list of availables trackers in openCV
vector<string> trackerTypes = {"boosting", "mil", "kcf", "tld", "medianflow", "goturn", "mosse", "csrt"}; 

/** \brief   Create a tracker according to the given name.
 *  \param[in] trackerType The name of the tracker to create.
 *  \returns The created tracker.
 */
cv::Ptr<cv::Tracker> createTrackerByName(string trackerType) {
    cv::Ptr<cv::Tracker> tracker;
    // This structure can be substituted with a more elegant switch in an early future.
    if (trackerType == trackerTypes[0])
        tracker = cv::TrackerBoosting::create();
    else if (trackerType == trackerTypes[1])
        tracker = cv::TrackerMIL::create();
    else if (trackerType == trackerTypes[2])
        tracker = cv::TrackerKCF::create();
    else if (trackerType == trackerTypes[3])
        tracker = cv::TrackerTLD::create();
    else if (trackerType == trackerTypes[4])
        tracker = cv::TrackerMedianFlow::create();
    else if (trackerType == trackerTypes[5])
        // GOTURN required that the caffe model file are placed in the root folder of the project...
        // There are no alternatives as stated here: https://stackoverflow.com/a/48984932/13239174
        tracker = cv::TrackerGOTURN::create();
    else if (trackerType == trackerTypes[6])
        tracker = cv::TrackerMOSSE::create();
    else if (trackerType == trackerTypes[7])
        tracker = cv::TrackerCSRT::create();
    else {
        cout << "Incorrect tracker name" << endl;
        cout << "Available trackers are: " << endl;
        for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it){
            std::cout << " " << *it << endl;
        }
    }
    return tracker;
}

/** \brief   Generate a random RGB color. Aka 3 values in the range [0, 255].
 *  \returns The generated color.
 */
Scalar generateRandomColor(){
    // NB: the Random Number Generator use a seed that depends on the time. Hence each execution of the code will have a different seed.
    // But a main problem is multiple calls across the same execution: the seed is the same because the time do not change so fast. 
    // Hence the variable is static meaning that only one initialization per code execution is required.
    Scalar color = Scalar(randUniform(0, 255), randUniform(0, 255), randUniform(0, 255));
    return color;
}

/** \brief   Fill the vector with random genereted colors.
 *  \param[out] colors The output vector that will store the generated colors.
 *  \param[in] numColors How many colors need to be generated.
 */
void generateRandomColors(vector<Scalar> & colors, int numColors){
    for(int i=0; i < numColors; i++){
        colors.push_back(generateRandomColor()); 
    }
}

/** \brief  A function that test the functionalities of input for the LoadVideo class and of output for the StoreVideo class.
 *  \return None.
 */
void testIO(){
    LoadVideo lv = LoadVideo("./videosIn/shelfy.mp4", 1.0, true); //./videosIn/dashcam.mp4
    StoreVideo sv = StoreVideo("videosOut/out", true);

    bool end = false;
    while(!end){
        Mat frame = lv.read();
        if(frame.empty()){
            cout << "end of video\n";
            break;
        } else{
            end = sv.addFrame(frame);
        }
        sleepMil(1000);
        cout << "fps: " << lv.fps() << endl;
    }
    cout << "end of while\n";
}


/** \brief  Main function of this test.
 *  \return None
*/
int main(int argc, char * argv[]) {
    // testIO();
    // return(0);

    // Set the tracker type. Change this to try different trackers.
    string trackerType = "csrt";

    cout << "multiTracker.cc\n\n";

    cout << "Default tracking algoritm is csrt" << endl;
    cout << "Available tracking algorithms are:" << endl;
    for (vector<string>::iterator it = trackerTypes.begin(); it != trackerTypes.end(); it++){
        std::cout << "\t" << *it << endl;
    }

    // set default values for tracking algorithm and video
    string videoPath = "./videosIn/dashcam.mp4";

    // create a video capture object to read videos
    LoadVideo lv = LoadVideo(videoPath, 1000, true);
    cout << "source fps: " << lv.sourceFPS() << endl;
    
    // create the class to save the video
    StoreVideo sv = StoreVideo("videosOut/out", true);

    // NB: this next few lines are repated (it does not matter in this small example).
    Mat frame = lv.read();
    // resize for faster processing
    downScaleImage(frame);

    // draw bounding boxes over objects
    // selectROI's default behaviour is to draw box starting from the center
    // when fromCenter is set to false, you can draw box starting from top left corner
    cout << "\n==========================================================\n";
    cout << "Press Space or Enter to exit selection process" << endl;
    cout << "\n==========================================================\n";

    // Initialize Tracker with tracking algorithm
    bool showCrosshair = true;
    bool fromCenter = false;
    Rect2d bbox = Rect2d(cv::selectROI("Object selection", frame, showCrosshair, fromCenter));
    destroyWindow("Object selection");  //the window of the selection is not automaically closed.

    // quit if there are no objects to track
    if(bbox.empty()){ 
        return 0;
    }

    // get a random color
    Scalar color = generateRandomColor(); 
    cout << "color: " << color << endl;

    // Start timer
    // the official documentation is at: https://docs.opencv.org/3.4/db/de0/group__core__utils.html#gae73f58000611a1af25dd36d496bf4487
    int processedFrames = 0;

    // Create multitracker
    cv::Ptr<cv::Tracker> tracker = createTrackerByName(trackerType);
    tracker->init(frame, bbox);  //rect2d works with double while rect works with int

    // process video and track objects
    cout << "\n==========================================================\n";
    cout << "Started the tracking, press 'q' to quit." << endl;

    // track until the videoCapture is available
    while(true) {
        // get frame from the video
        frame = lv.read();
        // stop the program if reached end of video
        if (frame.empty()) break;
        // resize for faster processing
        downScaleImage(frame);

        //update the tracking result with new frame
        bool ok = tracker->update(frame, bbox);

        // draw the tracked object
        if (ok){
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, color, 2, 1);
        } else{
            // Tracking failure detected.
            putText(frame, "Tracking failure", Point(10,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
        
        // Display tracker type on frame
        putText(frame, "Tracker: " + trackerType, Point(10,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
         
        // Display FPS on frame
        string fpsStr = doubleToString(lv.fps(), 1);
        putText(frame, "FPS : " + fpsStr, Point(10,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        bool quit = sv.addFrame(frame);
        if(quit){break;}
    }
return(0);
}

