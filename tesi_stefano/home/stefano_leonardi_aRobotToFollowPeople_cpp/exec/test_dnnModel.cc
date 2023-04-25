/** @file   test_dnnModel.cc
 *  @brief  A test file used for check the functionalities of the DnnModel class.
 *  @author Stefano Leonardi
 ***********************************************/
// usage: cls && ./compile.sh && ./build/buildLinux/dnnModels
#include "dnnModel.hh"
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>


/** \brief  Test function to execute a classifier instance. 
 *  \return None
*/
void testClassifier(){
    // load image
    Mat image = cv::imread("./imagesIn/samplePeople/soccer.jpg");
    std::unique_ptr<ClassifierModel> classifier = std::make_unique<GoogleNet>(); //GoogleNet or ResNet50

    vector<float> encoding;
    classifier->feed(image, encoding);

    watch(encoding.size());
    watch(encoding);
}

/** \brief  Test function to execute a detector instance.
 *  \param[in] peopleOnly If run normal detector or filter based on people.  
 *  \return None.
 */
void testDetector(bool peopleOnly=true){
    // load image
    string path = "./imagesIn/samplePeople/horse.jpg";
    Mat image = cv::imread(path);
    std::unique_ptr<DetectorModel> detector = std::make_unique<YOLOv3>(); // YOLOv3 or MobileNetSSD
    
    ///// any kind of box
    vector<vector<int> > predictions;
    if(peopleOnly){
        detector->detectPeopleOnly(image, predictions, 0.5);
    } else{
        detector->feed(image, predictions, 0.5);
    }
    detector->drawPredictions(predictions, image);
    
    // Create a window and show output
    string win;
    if(peopleOnly){
        win = "People only";
    } else{
        win = "Any box";
    }
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
    cv::imshow(win, image);
}

/** \brief  Main function of this test.
 *  \return None
*/
int main(int argc, char * argv[]){
    // Object ClassifierModel
    testClassifier();

    // Object DetectorModel
    testDetector(false);
    cv::waitKey();

    return(0);
}