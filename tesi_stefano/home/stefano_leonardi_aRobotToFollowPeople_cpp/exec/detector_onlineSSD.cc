/** @file   detector_onlineSSD.cc
 *  @brief  A working example of object detection with the SSD algorithm.
 *  @author Stefano Leonardi
 ***********************************************/
// usage: cls && ./compile.sh && ./build/buildLinux/detectorOnlineSSD
// source code: https://stackoverflow.com/questions/46728231/how-to-read-data-at-the-specific-coordinates-in-high-dimensional-matclass-using
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <iostream>
#include <string> 

using namespace cv;
using namespace cv::dnn;
using namespace std;

/** \brief  Function to create vector of names recognised by SSD.
 *  \return vector<String> The created vector.
 */
std::vector<String> createClaseNames(){
    std::vector<String> classNames;
    classNames.push_back("background");
    classNames.push_back("aeroplane");
    classNames.push_back("bicycle");
    classNames.push_back("bird");
    classNames.push_back("boat");
    classNames.push_back("bottle");
    classNames.push_back("bus");
    classNames.push_back("car");
    classNames.push_back("cat");
    classNames.push_back("chair");
    classNames.push_back("cow");
    classNames.push_back("diningtable");
    classNames.push_back("dog");
    classNames.push_back("horse");
    classNames.push_back("motorbike");
    classNames.push_back("person");
    classNames.push_back("pottedplant");
    classNames.push_back("sheep");
    classNames.push_back("sofa");
    classNames.push_back("train");
    classNames.push_back("tvmonitor");
    return classNames;
}  

/** \brief  Main function of this test.
 *  \return None
*/
int main(int argc, char **argv){
    // set inputs
    String modelTxt = "./models/mobileNet_SSD/MobileNetSSD_deploy.prototxt";
    String modelBin = "./models/mobileNet_SSD/MobileNetSSD_deploy.caffemodel";
    String imageFile = "./imagesIn/samplePeople/players.jpg";
    std::vector<String> classNames = createClaseNames();

    //read caffe model
    Net net;
    try{
        net = dnn::readNetFromCaffe(modelTxt, modelBin);
    }
    catch(cv::Exception& e){
        std::cerr << "Exception: " << e.what() << std::endl;
        if (net.empty()){
            std::cerr << "Can't load network." << std::endl;
            exit(-1);
        }
    }

    // read image 
    Mat img = imread(imageFile);
    Size imgSize = img.size();

    // create input blob
    Mat img300;
    resize(img, img300, Size(300, 300));
    Mat inputBlob = blobFromImage(img300, 0.007843, Size(300, 300), Scalar(127.5)); //Convert Mat to dnn::Blob image batch

    // apply the blob on the input layer
    net.setInput(inputBlob); //set the network input

    // classify the image by applying the blob on the net
    Mat detections = net.forward("detection_out"); //compute output

    // look what the detector found
    for (int i=0; i < detections.size[2]; i++) {

        // print information into console
        // cout << "-----------------" << endl;
        // cout << "Object nr. " << i + 1 << endl;

        // detected class
        int indxCls[4] = { 0, 0, i, 1 };
        int cls = detections.at<float>(indxCls);
        // std::cout << "class: " << classNames[cls] << endl;

        // confidence
        int indxCnf[4] = { 0, 0, i, 2 };
        float cnf = detections.at<float>(indxCnf);
        // std::cout << "confidence: " << cnf * 100 << "%" << endl;

        // bounding box
        int indxBx[4] = { 0, 0, i, 3 };
        int indxBy[4] = { 0, 0, i, 4 };
        int indxBw[4] = { 0, 0, i, 5 };
        int indxBh[4] = { 0, 0, i, 6 };
        int Bx = detections.at<float>(indxBx) * imgSize.width;
        int By = detections.at<float>(indxBy) * imgSize.height;
        int Bw = detections.at<float>(indxBw) * imgSize.width - Bx;
        int Bh = detections.at<float>(indxBh) * imgSize.height - By;
        // std::cout << "bounding box [x, y, w, h]: " << Bx << ", " << By << ", " << Bw << ", " << Bh << endl;

        // draw bounding box to image
        Rect bbox(Bx, By, Bw, Bh);
        rectangle(img, bbox, Scalar(255,0,255),1,8,0);

    }
    //show image
    String winName("image");
    imshow(winName, img);

    // Wait for keypress
    waitKey();

}