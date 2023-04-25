#include "videoIO.hh"

///////////////////////////   StoreVideo Class   /////////////////////////////////////////////
StoreVideo::StoreVideo(const string & dest, bool show, int fps) : show(show), fps(fps) {
    this->dest = dest + ".avi";
    // set the codec for the video writer
    this->fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G'); //MPEG or MJPG -> .avi
    this->toBeInitialized = true;
}

bool StoreVideo::addFrame(const Mat & frame){
    // initialize the video writer
    // if(! this->writer.has_value()){ //optional
    if(this->toBeInitialized){
        this->toBeInitialized = false;
        this->writer = VideoWriter(this->dest, this->fourcc, this->fps, frame.size(), true);
    }

    //store the given frame
    // this->writer.value().write(frame); //optional
    this->writer.write(frame);

    if(this->show){
        // show frame
        imshow("Frame", frame);

        // if the `q` key was pressed, break from the loop
        if (waitKey(1) == 'q'){
            return true;
        }
    }
    return false;
}

void StoreVideo::release(){
    // Release the file writer.
    // this->writer.value().release();  //optional
    this->writer.release();
}


///////////////////////////   LoadVideo Class   //////////////////////////////////////////////
LoadVideo::LoadVideo(const string & source, int warmUp, bool simulateRealtime) : \
source(source), warmUp(warmUp), simulateRealtime(simulateRealtime) {

    // check if manage realtime processing from a webcam or loading from file.
    this->realtime = isdigit(source[0]);
    if(this->realtime){
        this->webcam = atoi(source.c_str());
        this->cap.open(this->webcam);
        sleepMil(warmUp);
    } else{
        // example: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
        this->cap.open(this->source);

        // quit if unable to read the video file
        if(!cap.isOpened()){
            cout << "\n\n\nError opening video file " << this->source << "\n\n\n";
        }
    }
 
    // start counting the FPS rate
    this->frameProcessed = 0;

    // both should be true to start the simulate realtime processing
    this->simulateRealtime = simulateRealtime && !realtime;
    this->timeStart = -1;           //time of start processing time
    if(simulateRealtime){
        this->millisecPerFrame = 1000/sourceFPS(); //how many millisec each frame took
        this->totalFramesGone = 0; //total number of frames discarded and not processed
    }
}

int LoadVideo::sourceFPS(){
    return((int)this->cap.get(cv::CAP_PROP_FPS));
}

Mat LoadVideo::read(){
    Mat frame;
    // start the procedure to discard frames if needed from realtime simulation
    if(this->simulateRealtime){

        // measure how many frames have been processed compared to the expectations
        long long int timeElapsed = elapsed();
        int oldDiscard = this->totalFramesGone;      //total number of discarded frames
        this->totalFramesGone = (int)timeElapsed / this->millisecPerFrame;	//total number of frames to discard (from begin)

        int nFramesToDiscard = this->totalFramesGone - oldDiscard - 1; //how many frames left to burn
        // effectively discard the useless frames
        Mat discardedFrame;
        for(int i=0; i<nFramesToDiscard; i++){
            this->cap >> discardedFrame; //burn already gone frames
            if(discardedFrame.empty()){
                return discardedFrame;	//corner case: reach the end of the file video      
            }  
        }
    }

    // normal workflow of the code
    this->cap >> frame;
    this->frameProcessed++;
    return frame;
}

double LoadVideo::fps(){
    double fps = this->frameProcessed / (elapsed() / 1000.0);
    // cout << fps << " = " << frameProcessed << " / (" << elapsed() << " / 1000.0)\n\n";
    return fps;
}

long long int LoadVideo::elapsed(){
    // the first execution will set the start time
    if(this->timeStart == -1){
        this->timeStart = timeNow();
    }

    // measure the time gone since first execution
    long long int now = timeNow();
    long long int elapsed = (now - this->timeStart + 1);   //millisecond passed from the first frame captured
    return elapsed;
}

////////////////////////////   Utility   /////////////////////////////////////////////////
void downScaleImage(Mat & image, float downScale){
    // downscale the image by a certain ammout without changing the aspect ratio.
    cv::Size newSize = cv::Size((int)(image.size[1]*downScale), (int)(image.size[0]*downScale));
    cv::resize(image, image, newSize);
}

void downScaleImageTo(Mat & image, int width){
    // Size( newW, newW * oldH / oldW)
    cv::Size newSize = cv::Size(width, width * image.size[0] / image.size[1]); 
    cv::resize(image, image, newSize);
}

