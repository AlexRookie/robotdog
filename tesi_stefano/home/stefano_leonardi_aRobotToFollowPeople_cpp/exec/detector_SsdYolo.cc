/** @file   detector_SsdYolo.cc
 *  @brief  A working example of object detection with both SSD and YOLO algorithms.
 *  @author Stefano Leonardi
 ***********************************************/
// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
// cls && ./compile.sh && ./build/buildLinux/detectorSsdYolo --image=imagesIn/samplePeople/soccer.jpg
#include <fstream>
#include <sstream>
#include <iostream>
#include "utils.hh"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
;
using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
#define YOLO
#ifdef YOLO
vector<string> layerNames{"yolo_82", "yolo_94", "yolo_106"};
float confThreshold = 0.4; // Confidence threshold
float nmsThreshold = 0.6;  // Non-maximum suppression threshold
int resizeWidth = 416;  // Width of network's input image
int resizeHeight = 416; // Height of network's input image
#else
vector<string> layerNames{"detection_out"};
float confThreshold = 0.3; // Confidence threshold
int resizeWidth = 300;  // Width of network's input image
int resizeHeight = 300; // Height of network's input image
Size frameSize;
#endif
vector<string> classes;


/** \brief Remove the bounding boxes with low confidence using non-maxima suppression.
 *  \param[in] detections The image to be transformed with the predicted Boundig Boxes.
 *  \param[in] w The width of the input image.
 *  \param[in] h The height of the input image.
 *  \param[out] classIds The output vector that will store the ids.
 *  \param[out] confidences The output vector that will store the confidences.
 *  \param[out] bboxes The output vector that will store the bounding boxes.
 */
void postprocess(const vector<Mat> & detections, int w, int h, vector<int> & classIds, vector<float> & confidences, vector<Rect> & bboxes);
         
/** \brief Draw the bounding box prediction on the given image.
 *  \param[in] classId The id of the class prediction.
 *  \param[in] conf The confidence of the prediction.
 *  \param[in] left The left coordinate of the prediction.
 *  \param[in] top The top coordinate of the prediction.
 *  \param[in] right The right coordinate of the prediction.
 *  \param[in] bottom The bottom coordinate of the prediction.
 *  \param[in, out] frame The image ot be transformed.
 */
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat & frame);

/** \brief   Get the names of the output layers of the DNN.
 *  \param[in] net The network to work on.
 *  \returns The list of the names of the output layers.
 */
vector<String> getOutputsNames(const Net & net);

/** \brief  Main function of this test.
 *  \return None
*/
int main(int argc, char** argv){

    // process input parameters
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (!parser.has("image")){
        cout << "No image given. Stop.\n";
    } else{
        
        #ifdef YOLO
        cout << "\nRunning YOLO\n\n";
        // Load names of classes
        string classesFile = "./models/yoloV3-coco/coco.names";
        ifstream ifs(classesFile.c_str());
        // each line become an element of the vector
        string line;
        while (getline(ifs, line)) 
            classes.push_back(line);
        cout << "classes: " << classes << endl;

        // Give the configuration and weight files for the model
        String modelConf = "./models/yoloV3-coco/yolov3.cfg";
        String modelWeights = "./models/yoloV3-coco/yolov3.weights";
        // And Load the network
        Net net = readNetFromDarknet(modelConf, modelWeights);

        #else
        cout << "\nRunning SSD\n\n";
        classes = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
        string modelPath = "./models/mobileNet_SSD/";
        string modelTxt = modelPath + "MobileNetSSD_deploy.prototxt";
        string modelBin = modelPath + "MobileNetSSD_deploy.caffemodel";
        Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
        #endif
        
        // Open an image file and set the output name
        // For advance processing of video/webcam look at the source code
        string str, outputFile;
        VideoCapture cap;
        str = parser.get<String>("image");
        cap.open(str);
        #ifdef YOLO
        str.replace(str.end()-4, str.end(), "_yolo.jpg"); //replace extension
        #else
        str.replace(str.end()-4, str.end(), "_ssd.jpg"); //replace extension
        #endif
        outputFile = str;

        // Capture and process the frame.
        Mat frame;
        cap >> frame;
        Size frameSize = frame.size();
        cout << "frameSize: " << frameSize << endl;

        // Create a 4D blob from a frame.
        #ifdef YOLO
        Mat blob = blobFromImage(frame, 1/255.0, Size(resizeWidth, resizeHeight), Scalar(0,0,0));
        #else
        Mat blob = blobFromImage(frame, 2/255.0, Size(resizeWidth, resizeHeight), Scalar(127.5));
        #endif
        
        //Sets the input to the network
        net.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        // get the output of all the terminal layers: [yolo_82, yolo_94, yolo_106]
        // vector<string> layerNames = getOutputsNames(net); //the layer can name computed
        vector<Mat> detections;
        net.forward(detections, layerNames);
        cout << "detections.size(): " << detections.size() << endl;

        // Remove the bounding boxes with low confidence
        // Declare variables that will store the return values
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> bboxes;
        postprocess(detections, frameSize.width, frameSize.height, classIds, confidences, bboxes);
        
        // Show the prediction found
        for(int i=0; i<bboxes.size(); i++){
            Rect box = bboxes[i];
            int right = box.x + box.width;
            int bot = box.y + box.height;
            // cout << box.x << " - " << box.y << " - " << right << " - " << bot << endl;
            drawPred(classIds[i], confidences[i], box.x, box.y, right, bot, frame);
        }

        // Write the frame with the detection boxes
        imwrite(outputFile, frame);

        // Create a window and show output
        static const string kWinName = "Prediction";
        namedWindow(kWinName, WINDOW_AUTOSIZE);
        imshow(kWinName, frame);
        waitKey();
        cap.release();
    }

    return 0;
}


void postprocess(const vector<Mat> & detections, int w, int h, vector<int> & classIds, vector<float> & confidences, vector<Rect> & bboxes){
    cout << w << " " << h << endl;
    // For each mat produced by the DNN. The DNN produce a matrix for each output layer 3 in case of YOLO.
    cout << "\ndetections.size(): " << detections.size() << endl;
    for (int i=0; i < detections.size(); ++i){
        cout << "\tround: " << i+1 << "°\n";
        // Each produced matrix has R rows, one for each generated prediction (a lot has confidence value = 0)
        // and C=85 cols the first 5 elemnts are; [centerX, centerY, width, height, ???, ...] other 80 confidence values one for each class name. 

        // Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores. 
        // Assign the box's class label as the class with the highest score for the box.

        #ifdef YOLO
        cout << "detections[i] size: " << detections[i].rows << " - " << detections[i].cols << endl;
        float* data = (float*)detections[i].data; //a pointer to the first col of the actual row
        for (int j = 0; j < detections[i].rows; ++j, data += detections[i].cols){

            Mat scores = detections[i].row(j).colRange(5, detections[i].cols); // the rangeof confidence values:[5,85] -> it is a Mat
            Point classIdPoint;
            double confidence;
            // Locate the highest score in the "Mat" and retrive the classId (name) and the relative confidence value. This functon is like: argmax(scores).
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            // if confidence is high enough extract theuseful information
            if (confidence > confThreshold){
                int centerX = (int)(data[0] * w);
                int centerY = (int)(data[1] * h);
                int width = (int)(data[2] * w);
                int height = (int)(data[3] * h);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x); //I take the x vlaue (because we treat a vector but we are using a Mat datastructure)
                confidences.push_back((float)confidence);
                bboxes.push_back(Rect(left, top, width, height));
            }
        }
        #else
        // look what the detector found
        Mat detect = detections[i];
        for (int j=0; j < detect.size[2]; j++) {

            // confidence
            int indxCnf[4] = { 0, 0, j, 2 };
            float cnf = detect.at<float>(indxCnf);

            if(cnf > confThreshold){
                // detected class
                int indxCls[4] = { 0, 0, j, 1 };
                int cls = detect.at<float>(indxCls);

                // bounding box
                int indxBx[4] = { 0, 0, j, 3 };
                int indxBy[4] = { 0, 0, j, 4 };
                int indxBw[4] = { 0, 0, j, 5 };
                int indxBh[4] = { 0, 0, j, 6 };

                int left = (int)((detect.at<float>(indxBx)) * w);
                int top = (int)((detect.at<float>(indxBy)) * h);
                int width = (int)((detect.at<float>(indxBw)) * w) - left;
                int height = (int)((detect.at<float>(indxBh)) * h) - top;
                // cout << "bounding box [x, y, w, h]: " << left << ", " << top << ", " << width << ", " << height << endl;

                classIds.push_back(cls); //I take the x vlaue (because we treat a vector but we are using a Mat datastructure)
                confidences.push_back((float)cnf);
                bboxes.push_back(Rect(left, top, width, height));
            }
        }
        #endif
    }
    
    #ifdef YOLO
    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences (required only for YOLO)
    vector<int> indices; //output variable
    NMSBoxes(bboxes, confidences, confThreshold, nmsThreshold, indices);
    // preserve in the new vectors only the values that were not removed by NMS
    vector<int> new_classIds;
    vector<float> new_confidences;
    vector<Rect> new_bboxes;
    for (int i=0; i < indices.size(); ++i){
        int idx = indices[i];
        new_bboxes.push_back(bboxes[idx]);
        new_classIds.push_back(classIds[idx]);
        new_confidences.push_back(confidences[idx]);
    }
    bboxes = new_bboxes;
    classIds = new_classIds;
    confidences = new_confidences;
    #endif
}


void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame){
    Scalar color(255, 178, 50);
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), color, 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty()){
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
	putText(frame, label, Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
}


vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}