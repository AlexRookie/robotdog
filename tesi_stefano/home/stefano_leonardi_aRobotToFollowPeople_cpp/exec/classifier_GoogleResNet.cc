/** @file   classifier_GoogleResNet.cc
 *  @brief  A working example of object classifier with both GoogleNet and ResNet algorithms.
 *  @author Stefano Leonardi
 ***********************************************/
// usage: cls && ./compile.sh && ./build/buildLinux/classifierGoogleResNet

#include <fstream>
#include <sstream>
#include <iostream>
#include <string> 
#include <utils.hh>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

// #define ResNet50

// Initialize the parameters
#ifdef ResNet50
vector<string> layerNames{"OC2_DUMMY_0"};
string modelPath ="./models/resnet50-caffe2/resnet50-caffe2.onnx";
#else
vector<string> layerNames{"pool5/7x7_s1"};
string modelPath  = "./models/googleNet/bvlc_googlenet.prototxt";
string modelPath2 = "./models/googleNet/bvlc_googlenet.caffemodel";
#endif

/** \brief Print information of the given matrix.
 *  \param[in] m The matrix that need to be analysed.
 *  \param[in] name The name to be shown.
 */
void printMatDims(const Mat & m, const string name){
    // Print the dimension of the 4D matrix.
    cout << name << ".dims: " << m.dims << endl;
    cout << name << ".size[0] : " << m.size[0] << endl;
    cout << name << ".size[1] : " << m.size[1] << endl;
    cout << name << ".size[2] : " << m.size[2] << endl;
    cout << name << ".size[3] : " << m.size[3] << endl;
    cout << endl;
}

/** \brief Extract a sub portion of the entire 2D matrix according to the given parameters.
 *  \param[in] m The matrix that need to be processed.
 *  \param[in] dim The dimension to scan through. 
 *  \param[in] start The start index for the scan operation, in the given dimension.
 *  \param[in] end The end index for the scan operation, in the given dimension.
 */
vector<float> range1D(const Mat & m, int dim, int start, int end){
	// Extract a 1D matrix (aka a vector) from the given one ("m").
    // The vector is along the dimension ("dim"), from "start" to "end".
    vector<float> v;
    int idx[4] = {0, 0, 0, 0};
	for (int i=start; i<end; i++) {
        idx[dim] = i;
        v.push_back(m.at<float>(idx));
    }
	return v;
}

/** \brief  Main function of this test.
 *  \return None
*/
int main(int argc, char** argv){
    
    // load an input image
    string imagePath = "./imagesIn/samplePeople/soccer.jpg";
    Mat image = imread(imagePath);

    // Load the Network
    #ifdef ResNet50
    cout << "\nRunning ResNet50\n\n";
    Net net = readNetFromONNX(modelPath);
    #else
    cout << "\nRunning GoogleNet\n\n";
    Net net = readNetFromCaffe(modelPath, modelPath2);
    #endif

    // generate the blob
    Mat blob = blobFromImage(image, 1, Size(224, 224), Scalar(0.485, 0.456, 0.406));

    // set the input to the network
    net.setInput(blob);

    // generate the encodings
    vector<Mat> encodingsMat;
    net.forward(encodingsMat, layerNames);
    cout << "encodingsMat.size(): " << encodingsMat.size() << endl;
    printMatDims(encodingsMat[0], "encodingsMat");
    
    // process the result to extract the 1024/2048 values that define the encodings of the image
    vector<float> enc = range1D(encodingsMat[0], 1, 0, encodingsMat[0].size[1]);
    // watch(enc);
    watch(enc.size());

    cout << "THE END\n";
return(0);
}