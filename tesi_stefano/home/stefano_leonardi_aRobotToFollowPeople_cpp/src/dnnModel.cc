#include "dnnModel.hh"

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////   class DnnModel   ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
DnnModel::DnnModel(string modelPath1, string modelPath2, const vector<string> & layerNames, bool gpuBackend) : \
modelPath1(modelPath1), modelPath2(modelPath2), layerNames(layerNames){
    if(gpuBackend){
        useCuda();
    }
}

////////////// set functions
void DnnModel::useCuda(){
    // TODO: test
    // set CUDA as the preferable backend and target

    if(cv::cuda::getCudaEnabledDeviceCount() > 0){
        cout << "[INFO] setting preferable backend and target to GPU and CUDA...\n";
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else{ 
        cout << "[WARNING] tried to setup a GPU backend but no Nvidia GPU found, CPU will be used.\n";
    }
}

////////////// main flow functions
void DnnModel::setInput(const Mat & blob){
    this->net.setInput(blob);
}

void DnnModel::forward(vector<Mat> & detections){
    net.forward(detections, this->layerNames);
}

////////////// utility functions
void DnnModel::printMatDims(const Mat & m, const string name){
    // Print the dimension of the 4D matrix.
    cout << name << ".dims: " << m.dims << endl;
    cout << name << ".size[0] : " << m.size[0] << endl;
    cout << name << ".size[1] : " << m.size[1] << endl;
    cout << name << ".size[2] : " << m.size[2] << endl;
    cout << name << ".size[3] : " << m.size[3] << endl;
    cout << endl;
}

void DnnModel::range1D(const Mat & m, vector<float> & enc, int dim, int start, int end){
	// Extract a 1D matrix (aka a vector) from the given one ("m").
    // The vector is along the dimension ("dim"), from "start" to "end".
    int idx[4] = {0, 0, 0, 0};
	for (int i=start; i<end; i++) {
        idx[dim] = i;
        enc.push_back(m.at<float>(idx));
    }
}



///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////   CLASSIFIER   ////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
ClassifierModel::ClassifierModel(string modelPath1, string modelPath2, const vector<string> & layerNames, bool gpuBackend) : \
DnnModel(modelPath1, modelPath2, layerNames, gpuBackend) {}

void ClassifierModel::blob(const Mat & image, Mat & blob){
    blob = blobFromImage(image, 1, Size(224, 224), Scalar(0.485, 0.456, 0.406));
}

void ClassifierModel::feed(const Mat & image, vector<float> & output){

    // This is a one time code used to test the performances of the different steps of the DNN models
    // if(false){
    //     static int rounds = 0;
    //     long long int startTime, blobTime, inputTime, forwardTime, processTime;
    //     static long long int blobSum, inputSum, forwardSum, processSum;
    //     startTime = timeNow();
    //     rounds++;

    //     std::cout << std::endl;
    //     Mat imageBlob;
    //     blob(image, imageBlob);
    //     blobTime = timeNow() - startTime;
    //     std::cout << "actual blobTime:    " << blobTime << std::endl;
    //     startTime = timeNow();

    //     setInput(imageBlob);
    //     inputTime = timeNow() - startTime;
    //     std::cout << "actual inputTime:   " << inputTime << std::endl;
    //     startTime = timeNow();

    //     vector<Mat> detections;
    //     forward(detections);
    //     forwardTime = timeNow() - startTime;
    //     std::cout << "actual forwardTime: " << forwardTime << std::endl;
    //     startTime = timeNow();

    //     Size frameSize = image.size();
    //     processDnnOutput(detections, output);
    //     processTime = timeNow() - startTime;
    //     std::cout << "actual processTime: " << processTime << std::endl;


    //     blobSum += blobTime;
    //     std::cout << "Classifier blob average:    " << blobSum*1.0/rounds << std::endl;
    //     inputSum += inputTime;
    //     std::cout << "Classifier input average:   " << inputSum*1.0/rounds << std::endl;
    //     forwardSum += forwardTime;
    //     std::cout << "Classifier forward average: " << forwardSum*1.0/rounds << std::endl;
    //     processSum += processTime;
    //     std::cout << "Classifier process average: " << processSum*1.0/rounds << std::endl;
    // }
    // else{
        Mat imageBlob;
        blob(image, imageBlob);
        setInput(imageBlob);
        
        vector<Mat> detections;
        forward(detections);
        
        processDnnOutput(detections, output);
    // }
}

void ClassifierModel::processDnnOutput(const vector<Mat> & detections, vector<float> & output){
    // process the result to extract the 1024/2048 values that define the encodings of the image
    range1D(detections[0], output, 1, 0, detections[0].size[1]);
}

///////////////////////   ResNet50   ////////////////////////////////////////////////////
ResNet50::ResNet50(bool gpuBackend, string modelPath1, const vector<string> & layerNames) : ClassifierModel(modelPath1, "", layerNames, gpuBackend) {
    this->net = readNetFromONNX(modelPath1);
}

///////////////////////   GoogleNet   ////////////////////////////////////////////////////
GoogleNet::GoogleNet(bool gpuBackend, string modelPath1, string modelPath2, const vector<string> & layerNames) : ClassifierModel(modelPath1, modelPath2, layerNames, gpuBackend) {
    this->net = readNetFromCaffe(modelPath1, modelPath2);
}


const std::string GoogleNetTRT::DEPLOY_ENGINE    = "models/googleNet/deploy.engine";
const cv::Scalar  GoogleNetTRT::PIXEL_MEANS      = cv::Scalar(104.f, 117.f, 123.f);
const int         GoogleNetTRT::ENGINE_SHAPE0[3] = {3, 224, 224};
const int         GoogleNetTRT::ENGINE_SHAPE1[3] = {1024, 1, 1}; //{1000, 1, 1};
const cv::Size    GoogleNetTRT::RESIZED_SHAPE    = cv::Size(224, 224);
const bool        GoogleNetTRT::doCropping       = true;


GoogleNetTRT::GoogleNetTRT() : ClassifierModel("", "", {}, true) {
    #ifdef NO_CUDA
        throw std::runtime_error("Cannot use TensorRT when CUDA support is disabled in CMakeLists.txt");
    #else
        init();
    #endif
}

void GoogleNetTRT::feed(const cv::Mat & image, std::vector<float> & output) {
    // normalize image
    cv::Mat nI = normalizeImg(image);
    
    // HWC -> CHW
    std::vector<float> dataIn;
    hwcToChw(nI, dataIn);
    
    // forward to network
    output.resize(ENGINE_SHAPE1[0]*ENGINE_SHAPE1[1]*ENGINE_SHAPE1[2]);

    //auto t0 = timeNow();
    #ifndef NO_CUDA
        net.forward(dataIn.data(), output.data());
    #endif
    //auto t1 = timeNow()-t0;
    //std::cerr << t1 << "ms" << std::endl;
}


void GoogleNetTRT::init() {
    #ifndef NO_CUDA
        net.initEngine(DEPLOY_ENGINE, const_cast<int*>(ENGINE_SHAPE0), const_cast<int*>(ENGINE_SHAPE1));
    #endif
}


cv::Mat GoogleNetTRT::normalizeImg(cv::Mat const & img) {
    cv::Mat crop = img;
    if (doCropping) {
        int h = img.rows;
        int w = img.cols;        
        if (h < w)
            crop = img(cv::Rect((w-h)/2, 0, h, h));
        else
            crop = img(cv::Rect(0, (h-w)/2, w, w));
    }

    // preprocess the image crop
    cv::resize(crop, crop, RESIZED_SHAPE);
    cv::Mat cropT;
    crop.convertTo(cropT, CV_32FC3);
    crop = cropT - PIXEL_MEANS;
    return crop;
}


void GoogleNetTRT::hwcToChw(cv::Mat const & src, std::vector<float> & data) {
    static const int INPUT_H = RESIZED_SHAPE.height;
    static const int INPUT_W = RESIZED_SHAPE.width;

    data.resize(ENGINE_SHAPE0[0]*ENGINE_SHAPE0[1]*ENGINE_SHAPE0[2]);
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            const cv::Vec3f& c = src.at<cv::Vec3f>(i,j);       
            data[0*INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(c[0]);
            data[1*INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(c[1]);
            data[2*INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(c[2]);
        }
    }
}                                                                               


///////////////////////   OsNet       ////////////////////////////////////////////////////
OsNet::OsNet(bool gpuBackend, string modelPath1, const vector<string> & layerNames) : ClassifierModel(modelPath1, "", layerNames, gpuBackend) {
    this->net = readNetFromONNX(modelPath1);
}

void OsNet::blob(const Mat & image, Mat & blob) {
    using namespace cv;

    Mat processed;

    // resize
    resize(image, processed, Size(128, 256));

    // BGR -> RGB
    cvtColor(processed, processed, COLOR_BGR2RGB);
  
    // normalize to float in [0,1]
    processed.convertTo(processed, CV_32F, 1.0 / 255);
  
    // normalization per channel
    Mat channels[3];
    split(processed, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    merge(channels, 3, processed);

    // HWC to CHW
    dnn::blobFromImage(processed, blob); 
}


///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////   DETECTOR   //////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
DetectorModel::DetectorModel(string modelPath1, string modelPath2, string namesPath, const vector<string> & layerNames, bool gpuBackend) : \
DnnModel(modelPath1, modelPath2, layerNames, gpuBackend), namesPath(namesPath) {
    // read the names of the classes and save them
    idClassPerson = -1;
    ifstream file(namesPath);
    if (!file){
        cout << "\n\nError unable to open the file of names.\n";
    } else{
        string line;
        // read line by line
        int i=0;
        while(getline(file, line)){
            // santitize input (Windows/Linux)
            line = trim(line);
            classes.push_back(line);
            // look for the ID of the class "person"
            if(line == "person"){
                idClassPerson = (int)classes.size()-1;
            }
        }
    }
}

void DetectorModel::feed(const Mat & image, vector<vector<int> > & output, float confidenceThresh){

    // This is a one time code used to test the performances of the different steps of the DNN models
    // if(false){
    //     static int rounds = 0;
    //     long long int startTime, blobTime, inputTime, forwardTime, processTime;
    //     static long long int blobSum, inputSum, forwardSum, processSum;
    //     startTime = timeNow();
    //     rounds++;

    //     std::cout << std::endl;
    //     Mat imageBlob;
    //     blob(image, imageBlob);
    //     blobTime = timeNow() - startTime;
    //     std::cout << "actual blobTime:    " << blobTime << std::endl;
    //     startTime = timeNow();

    //     setInput(imageBlob);
    //     inputTime = timeNow() - startTime;
    //     std::cout << "actual inputTime:   " << inputTime << std::endl;
    //     startTime = timeNow();

    //     vector<Mat> detections;
    //     forward(detections);
    //     forwardTime = timeNow() - startTime;
    //     std::cout << "actual forwardTime: " << forwardTime << std::endl;
    //     startTime = timeNow();

    //     Size frameSize = image.size();
    //     processDnnOutput(detections, output, frameSize, confidenceThresh);
    //     processTime = timeNow() - startTime;
    //     std::cout << "actual processTime: " << processTime << std::endl;


    //     blobSum += blobTime;
    //     std::cout << "Detector blob average:    " << blobSum*1.0/rounds << std::endl;
    //     inputSum += inputTime;
    //     std::cout << "Detector input average:   " << inputSum*1.0/rounds << std::endl;
    //     forwardSum += forwardTime;
    //     std::cout << "Detector forward average: " << forwardSum*1.0/rounds << std::endl;
    //     processSum += processTime;
    //     std::cout << "Detector process average: " << processSum*1.0/rounds << std::endl;
    // }
    // else{
        // Normal execution:
        Mat imageBlob;
        blob(image, imageBlob);
        setInput(imageBlob);

        vector<Mat> detections;
        forward(detections);

        Size frameSize = image.size();
        processDnnOutput(detections, output, frameSize, confidenceThresh);
    // }
}

void DetectorModel::detectPeopleOnly(const Mat & image, vector<vector<int> > & output, float confidenceThresh){
    vector<vector<int> > tmp;
    feed(image, tmp, confidenceThresh);

    // filter out the detections that do not belong to the class "person"
    for(int i=0; i<tmp.size(); i++){
        if(tmp[i][CLASSID] == idClassPerson){
            output.push_back(tmp[i]);
        }
    }
}


void DetectorModel::drawOnePrediction(const vector<int> & detection, Mat & frame, const Scalar & color, bool showConfidence){
    //Draw a rectangle displaying the bounding box
    int left = detection[LEFT];
    int top = detection[TOP];
    int right  = (left + detection[WIDTH]);
    int bottom = (top + detection[HEIGHT]);
    rectangle(frame, Point(left, top), Point(right, bottom), color, 2);
    
    if(showConfidence){
        //Get the label for the class name and its confidence
        string label = cv::format("%.2f", detection[CONFIDENCE]/100.0);
        if (!classes.empty()){
            label = classes[detection[CLASSID]] + ": " + label + "%";
        }

        //Display the label at the top of the bounding box
        int baseLine;
        Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = std::max(top, labelSize.height);
        putText(frame, label, Point(left, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
}

void DetectorModel::drawPredictions(const vector<vector<int> > & detections, Mat & frame, const Scalar & color, bool showConfidence){
    for(int i=0; i<detections.size(); i++){
        drawOnePrediction(detections[i], frame, color, showConfidence);
    }
}


///////////////////////   MobileNetSSD   //////////////////////////////////////////////////
MobileNetSSD::MobileNetSSD(string modelPath1, string modelPath2, string namesPath, const vector<string> & layerNames, bool gpuBackend) : DetectorModel(modelPath1, modelPath2, namesPath, layerNames, gpuBackend) {
    this->net = readNetFromCaffe(modelPath1, modelPath2);
}

void MobileNetSSD::blob(const Mat & image, Mat & blob){
    blob = blobFromImage(image, 2/255.0, Size(300, 300), Scalar(127.5));
}

void MobileNetSSD::processDnnOutput(const vector<Mat> & detections, vector<vector<int> > & output, const Size & size, float confidenceThresh){
    
    // each output layer produce one instance of this vector.
    // SSD use one output layer hence this "for" will execute one loop, in case of default.
    for (int i=0; i < detections.size(); i++){

        // look what the detector had found
        Mat detection = detections[i];
        for (int j=0; j < detection.size[2]; j++) {

            // confidence
            int indxCnf[4] = { 0, 0, j, 2 };
            // note that I should extract a float and then cast to float. If I directly extract a float I will extract a different number...
            float confidence = detection.at<float>(indxCnf);
            if(confidence > confidenceThresh){

                // detected class
                int indxCls[4] = { 0, 0, j, 1 };
                int cls = (int)detection.at<float>(indxCls);

                // bounding box
                int indxBx[4] = { 0, 0, j, 3 };
                int indxBy[4] = { 0, 0, j, 4 };
                int indxBw[4] = { 0, 0, j, 5 };
                int indxBh[4] = { 0, 0, j, 6 };

                // force the BB to be inside the frame size
                int left = std::max(0, (int)(detection.at<float>(indxBx) * size.width));
                int top = std::max(0, (int)(detection.at<float>(indxBy) * size.height));
                int width  = std::min(size.width , (int)(detection.at<float>(indxBw) * size.width )) - left;
                int height = std::min(size.height, (int)(detection.at<float>(indxBh) * size.height)) - top;

                // prepare the computed values for the return
                vector<int> detectionVect = {cls, (int)(confidence*10000.0), left, top, width, height};
                output.push_back(detectionVect);
            }
        }
    }
}



///////////////////////   YOLOv3   //////////////////////////////////////////////////
YOLOv3::YOLOv3(string modelPath1, string modelPath2, string namesPath, const vector<string> & layerNames, bool gpuBackend, float confidenceNMS) : DetectorModel(modelPath1, modelPath2, namesPath, layerNames, gpuBackend), confidenceNMS(confidenceNMS) {
    this->net = readNetFromDarknet(modelPath1, modelPath2);
}

void YOLOv3::blob(const Mat & image, Mat & blob){
    blob = blobFromImage(image, 1/255.0, Size(416, 416), Scalar(0,0,0));
}

void YOLOv3::processDnnOutput(const vector<Mat> & detections, vector<vector<int> > & output, const Size & size, float confidenceThresh){
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> bboxes;
    // for each generated detecion
    for (int i=0; i < detections.size(); ++i){

        float* data = (float*)detections[i].data; //a pointer to the first col of the actual row
        for (int j = 0; j < detections[i].rows; j++, data += detections[i].cols){

            Mat scores = detections[i].row(j).colRange(5, detections[i].cols); // the rangeof confidence values:[5,85] -> it is a Mat

            Point classIdPoint;
            double confidence;
            // Locate the highest score in the "Mat" and retrive the classId (name) and the relative confidence value. This functon is like: argmax(scores).
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            // if confidence is high enough extract the useful information
            if (confidence > confidenceThresh){
                int centerX = (int)(data[0] * size.width); 
                int centerY = (int)(data[1] * size.height); 
                int width  = (int)(data[2] * size.width ); 
                int height = (int)(data[3] * size.height); 
                // force the BB to be inside the frame size
                int left = std::max(0, centerX - width  / 2);
                int top  = std::max(0, centerY - height / 2);
                width  = std::min(size.width - left, width );
                height = std::min(size.height - top, height);

                classIds.push_back(classIdPoint.x); //I take the x vlaue (because we treat a vector but we are using a Mat datastructure)
                confidences.push_back((float)confidence);
                bboxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences (required only for YOLO)
    vector<int> indices; //output variable
    NMSBoxes(bboxes, confidences, confidenceThresh, confidenceNMS, indices);
    // preserve in the new vectors only the values that were not removed by NMS
    for (int i=0; i < indices.size(); ++i){
        int idx = indices[i];
        vector<int> detectionVect = {classIds[idx], (int)(confidences[idx]*10000.0), bboxes[idx].x, bboxes[idx].y, bboxes[idx].width, bboxes[idx].height};
        output.push_back(detectionVect);
    }
}
