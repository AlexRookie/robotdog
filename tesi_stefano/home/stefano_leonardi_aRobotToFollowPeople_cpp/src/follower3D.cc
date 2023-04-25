#include "follower3D.hh"

#include <jsoncons/json.hpp>
#include <jsoncons_ext/cbor/cbor.hpp>
#include <jsoncons_ext/jsonpath/jsonpath.hpp>


static const double DST_FALSE_NEGATIVE = 0.4;
static const double DST_FALSE_POSITIVE = 0.4;


using namespace jsoncons; // for convenience

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////   Constructor section   ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
Follower3D::Follower3D(DetectorNames detectorName, TrackerNames trackerName, ClassifierNames classifierName, bool useCuda, string negativePeopleDataset) : useCuda(useCuda), negativePeopleDataset(negativePeopleDataset) {
    setLogParam();
    setHyperParam();

    // Set default additional variables
    defaultPosition = Rect2d(-1, -1, 0, 0); 
    breakPosition = Rect2d(-2, -2, 0, 0);   
    flagPhase1 = true;
    // color in the BGR format of openCV
    colorPhase1Leader   = Scalar(255, 0, 0);        // blue
    colorPhase1Someone  = Scalar(150, 150, 150);    // gray
    colorPhase2Leader   = Scalar(0, 255, 0);        // green
    colorPhase2LeaderRejected   = Scalar(0, 80, 0); // dark green
    colorPhase2Someone  = Scalar(0, 0, 255);        // red
    colorTracking       = Scalar(0, 0, 0);          // black
    colorDrift          = Scalar(0, 255, 255);      // yellow
    // drift optimization
    this->lastWidth = 100;
    this->startDrifting = timeNow();

    // Load objDetector
    switch(detectorName){
        case SSD:       detector = make_unique<MobileNetSSD>(); break;
        case YOLO:    detector = make_unique<YOLOv3>();       break;
        default:        cout << "[INFO] Warning: no detector were chosen...\n";
    }
    if(useCuda){detector->useCuda();}

    // Load objTracker
    this->trackerName = trackerName;

    // Load imgClassifier
    switch(classifierName){
        case RESNET50:      classifier = make_unique<ResNet50>();   break;
        case GOOGLENET:     classifier = make_unique<GoogleNet>();  break;
        case GOOGLENET_TRT: classifier = make_unique<GoogleNetTRT>();  break;
        case OSNET:         classifier = make_unique<OsNet>();  break;
        default:        cout << "[INFO] Warning: no classifier were chosen...\n";
    }
    if(useCuda){classifier->useCuda();}

    // Create KNN instance
    knn = make_unique<KNN>();

    // Init input video
    videoIn = make_unique<LoadVideo3D>(1000);
}

void Follower3D::setLogParam(bool showLog, string destVideo, int downscaleWidth, int destFps){
    this->showLog = showLog;
    this->downscaleWidth = downscaleWidth;
    if(showLog){
        videoOut = make_unique<StoreVideo>(destVideo, true, destFps);
    }
}

void Follower3D::setHyperParam(int len_slowStartPhase, float confidenceThresh, int detectionFreq, int k, float driftRatio, int driftTollerance){
    this->len_slowStartPhase = len_slowStartPhase;
    this->confidenceThresh = confidenceThresh;
    this->detectionFreq = detectionFreq;
    this->k = k;
    this->driftRatio = driftRatio;
    this->driftTollerance = driftTollerance;
}

void Follower3D::createTracker(){

    cv::TrackerCSRT::Params csrtParams;
    csrtParams.admm_iterations = 2;
    csrtParams.template_size = 100;
    //std::cout << "template_size: " << csrtParams.template_size << std::endl;

    switch(trackerName){
        case CSRT:      tracker = cv::TrackerCSRT::create(csrtParams);        break;
        case BOOSTING:  tracker = cv::TrackerBoosting::create();    break;
        case MIL:       tracker = cv::TrackerMIL::create();         break;
        case KCF:       tracker = cv::TrackerKCF::create();         break;
        case TLD:       tracker = cv::TrackerTLD::create();         break;
        case MEDIANFLOW:tracker = cv::TrackerMedianFlow::create();  break;
        case GOTURN:    tracker = cv::TrackerGOTURN::create();      break;
        case MOSSE:     tracker = cv::TrackerMOSSE::create();       break;
        default:        cout << "[INFO] Warning: no tracker were chosen...\n";
    }
}


///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////   Follow section   ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
TimedPose Follower3D::follow(bool est_pose, double est_x_h, double est_y_h){
    TimedPose result;
    Rect2d leaderPosition;
    FrameData3D frame = videoIn->read();
    
    // corner case (no input from source)
    if(frame.frame.empty()){
        cout << "[INFO] End of video source\n";
        leaderPosition = breakPosition;
    
    } else{
        // check how much time has gone, up to now (if enough switch to phase 2)
        long long int elapsedMill = videoIn->elapsed();
        if(elapsedMill > len_slowStartPhase){
            flagPhase1 = false;
        }

        if(flagPhase1){ //phase 1: slowStart
            slowStartPhase(frame, leaderPosition);
        } else{         //phase 2: leaderTracking
            leaderTrackingPhase(frame, leaderPosition, est_pose, est_x_h, est_y_h);
        }
    }

    if (leaderPosition != breakPosition && leaderPosition != defaultPosition) {
        result.valid = bbox2World(frame, leaderPosition, result.x, result.y);
        result.timestamp = frame.lastTime;
    }

    // const int SZ = 800;
    // const float VMIN = -5;
    // const float VMAX =  5;
    // const float SV = SZ/(VMAX-VMIN);
    // cv::Mat img(SZ, SZ, CV_8UC3, cv::Scalar(0,0,0));
    
    // for(auto& v : hist) {
    //     cv::circle(img, cv::Point((v.first-VMIN)*SV, (v.second-VMIN)*SV), 10, cv::Scalar(255,0,0), -1);
    // }
    // cv::imshow("track", img);


    // update the last position if it is different from the default
    if(leaderPosition != defaultPosition){
        this->lastKnownPosition = centerBB(leaderPosition);
    }
    return result;
}



void Follower3D::slowStartPhase(FrameData3D & frame, Rect2d & leaderPosition){

    // look for detections in the image
    vector<vector<int> > detections;
    detector->detectPeopleOnly(frame.frame, detections, confidenceThresh);
    
    // By assumption in this phase the biggest BB (bounding box) represent the leader
    // Hence, we search for the index of the biggest BB
    int maxArea = 0, maxIndex = -1, area;
    for(int i=0; i<detections.size(); i++){
        area = detections[i][4] * detections[i][5]; //width*height
        if(area > maxArea){
            maxArea = area;
            maxIndex = i;   //aka the index of the leader
        }
    }

    // set the leader BB to be returned
    if(maxIndex==-1){
        leaderPosition = defaultPosition;
    } else{
        leaderPosition = Rect2d(detections[maxIndex][LEFT], detections[maxIndex][TOP], detections[maxIndex][WIDTH], detections[maxIndex][HEIGHT]); //centerBB(detections[maxIndex][LEFT], detections[maxIndex][TOP], detections[maxIndex][WIDTH], detections[maxIndex][HEIGHT]);
    }

    // add the BB to KNN according to their labels and eventually if needed a negative sample is loaded from the NegativePeople dataset
    addBBsToKNN(frame.frame, detections, maxIndex);

    // NB: it is fundamental to draw on the frame AFTER that the classifier has elaborated the frame itself, or, in case of elaboration before, is neccessary to draw on a COPY of the frame.
	// If this do not happen the classifier works on the drawn frame, and it will learn the "user friendly draw" and not the frame itself
    if(showLog){ //show visual information if required
        // draw the predictions. the leader will have a different color
        drawDetectionsToFrame(frame.frame, detections, colorPhase1Someone, maxIndex, colorPhase1Leader);

        bool quit = showFrame(frame.frame);
        if(quit){
            leaderPosition = breakPosition;
        }
    }
}


void Follower3D::leaderTrackingPhase(FrameData3D & frame, Rect2d & leaderPosition, bool est_pose, double est_x_h, double est_y_h){
    leaderPosition = defaultPosition; //standard return value

    // Guarantee that a detection is performed once every 'detectionFreq' milliseconds.
    static int detectionRounds = 0; //how many detection have been effectively done
    long long int elapsedMill = videoIn->elapsed() - len_slowStartPhase; //time since start of phase 2
    // compute: how many detection has been already performed according to the time gone.
    int detectionsEstimated = (int)(elapsedMill / detectionFreq);

    if(detectionRounds<=detectionsEstimated){   //perform OBJECT DETECTION
        //>>>>> calculate the drift radius tollerance according to the ratio, the period gone and the BB size
        long long int driftPeriod = timeNow() - this->startDrifting + 1; //milliseconds since the instantiation of the tracking
        // the formula is: d = t*r*w^2 + k
        // where: t is the time passed, r is a custom ratio, w^2 simulate the depth (if big the depth is low and d grows fast), k is a limit if t is close to zero
        float driftRadius = (float)(driftPeriod * this->driftRatio * pow(this->lastWidth/50.0, 2) + this->driftTollerance);

        //>>>>> look for detections in the image
        vector<vector<int> > detections;
        detector->detectPeopleOnly(frame.frame, detections, confidenceThresh);
        
        //>>>>> generate the encoding for each detection
        int len = (int)detections.size();
        vector<vector<float> > encodings(len);
        vector<int> labels(len);
        vector<double> dists(len);

        vector<int> leaderIndexes;
        vector<int> cnt(len);
        for(int i=0; i<len; i++){
            Rect rectBB = Rect(detections[i][2], detections[i][3], detections[i][4], detections[i][5]);
            // For the future: The KNN performances could be improved by removing the backgroud from the BB. An idea is to preserve only the central area of the BB. For example by cropping out an ellipse. (a not easy task: https://stackoverflow.com/questions/5207147/opencv-crop-image-with-ellipse)
            // crop only the intresting area of the image (AKA the BB)
            Mat cropBB = frame.frame(rectBB);

            classifier->feed(cropBB, encodings[i]);
                        				
            // filter out detection too far away according to the drift radius
			float drift = (float)cv::norm(this->lastKnownPosition - centerBB(rectBB)); //example: https://stackoverflow.com/a/28991348/13239174
            //if(drift > driftRadius){
                //labels[i] = SOMEONE;    // detections too far away are automatically rejected
            //} else{
            // use KNN to choose the right class of detections that are "close"
            labels[i] = knn->predict(encodings[i], dists[i], cnt[i], k);
            //}

            std::cout << "Number of positive cnt:\n";
            for (int i=0; i<len; i++) std::cout << labels[i] << " " << cnt[i] << " " << dists[i] << "\n";


            //if (labels[i]==LEADER && dists[i]>400) { // && dists[i]<0) {
              //  labels[i] = UNKNOWN;
            //} 

            // CHECK WITH EXPECTED POSE
            if (est_pose) {
                // get camera coordinates of i^th detection
                float c_x, c_y;
                bool valid = bbox2World(frame, rectBB, c_x, c_y);
                // check distance and filter
                if (valid) {
                    double dst = hypot(c_x-est_x_h, c_y-est_y_h);

                    // avoid false negatives
                    if ((labels[i]==SOMEONE || labels[i]==UNKNOWN) && dst<DST_FALSE_NEGATIVE) {
                        labels[i] = UNKNOWN; //LEADER; //UNKNOWN;
                        std::cout << "PROMOTED TO LEADER" << std::endl;
                    }
                    else if ((labels[i]==LEADER || labels[i]==UNKNOWN) && dst>DST_FALSE_POSITIVE) {
                        labels[i] = UNKNOWN; //SOMEONE; //UNKNOWN;
                        std::cout << "REJECTED AS LEADER" << std::endl;
                    }
                }
            }

            if(labels[i]==LEADER){
                leaderIndexes.push_back(i);
            }
        }
        
        //>>>>> If exist more than one leader I choose the one closer to the robot (AKA the one with the largest bounding box)
        int leaderIdx = -1;
        if(leaderIndexes.size()>=1){

            // compute the areas to choose the leader
            //int maxArea = 0, area;
            double minDist = 9999999999, dist;
            for(int l=0; l<leaderIndexes.size(); l++){
                int idx = leaderIndexes[l];
                //area = detections[idx][4] * detections[idx][5]; //width*height
                dist = dists[l]; 
                if(dist < minDist){
                    minDist = dist;
                    leaderIdx = l;   //aka the index of the leader
                }
                // if(area > maxArea){
                //     maxArea = area;
                //     leaderIdx = l;   //aka the index of the leader
                // }
            }
            // set all the "small" leaders as SOMEONE (note that: the two for-loop can be collapsed but this layout is clearer)
            for(int l=0; l<leaderIndexes.size(); l++){
                int idx = leaderIndexes[l];
                if(l!=leaderIdx){
                    labels[idx] = SOMEONE; //this fake leader is set as someone
                }
            }

            //>>>>> Initialize the object tracker
            detectionRounds++; //the detection has been performed correctly
            int leaderDetecIdx = leaderIndexes[leaderIdx]; 
            Rect2d leaderBB = Rect2d(detections[leaderDetecIdx][LEFT], detections[leaderDetecIdx][TOP], detections[leaderDetecIdx][WIDTH], detections[leaderDetecIdx][HEIGHT]);
            createTracker();
            tracker->init(frame.frame, leaderBB);

            //>>>>> set the values for the drift optimization
            this->startDrifting = timeNow();  //for all the duration of tracking the BB can drift away
            this->lastWidth = (int)leaderBB.width;

        } else{
            //corner case: multiple detection fails and the detectionRounds are not increased (hence force the detectionRounds to have a not too low value)
            if(detectionRounds<detectionsEstimated){
                detectionRounds = detectionsEstimated - 1;
            }
        }

        //>>>>> compute the leader position to be returned
        if(leaderIdx!=-1){
            int leaderDetecIdx = leaderIndexes[leaderIdx]; 
            leaderPosition = Rect2d(detections[leaderDetecIdx][2], detections[leaderDetecIdx][3], detections[leaderDetecIdx][4], detections[leaderDetecIdx][5]);
        }

        //>>>>> Add the detections (now with at most one leader) to KNN (AKA train KNN)
        // TODO: temporarily FREEZING KNN
        //knn->train(encodings, labels);
        //if(leaderIndexes.size()>0 && encodings.size()==1){
        //    addNegativeToKNN(); // AKA I have only the leader, then I choose a negative to improve the KNN space
        //}

        //>>>>> output visualization
        if(showLog){
            // draw BBs for leader and someone
            drawDetectionsToFrame(frame.frame, detections, colorPhase2Someone, ((leaderIdx==-1) ? -1 : leaderIndexes[leaderIdx]), colorPhase2Leader);

            // re-color the rejected leader with a dark green
            vector<vector<int> > tmp_rejectedLeader;
            for(int l=0; l<leaderIndexes.size(); l++){
                int idx = leaderIndexes[l];
                if(l!=leaderIdx){
                    tmp_rejectedLeader.push_back(detections[idx]); //this fake leader is set as someone
                }
            }
            // draw BBs for leader and rejected leaders
            drawDetectionsToFrame(frame.frame, tmp_rejectedLeader, colorPhase2LeaderRejected);

            // re-color the UNKNOWN detections with gray
            vector<vector<int> > tmp_unknown;
            for(int l=0; l<labels.size(); l++){
                if(labels[l]==UNKNOWN){
                    tmp_unknown.push_back(detections[l]); //this UNKNOWN is set as someone
                }
            }
            // draw BBs for UNKNOWN
            drawDetectionsToFrame(frame.frame, tmp_unknown, cv::Scalar(100,100,100));

            // draw leader KNN average distance from its class
            for(int l=0; l<leaderIndexes.size(); l++){
                int idx = leaderIndexes[l];
                
                string label = cv::format("%.2f", dists[l]);
            
                //Display the label at the bottom of the bounding box
                int left = detections[idx][LEFT];
                int bottom = detections[idx][TOP]+detections[idx][HEIGHT];
                int baseLine;
                Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                //bottom = std::max(top, labelSize.height);
                putText(frame.frame, label, Point(left, bottom - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 2);
            }            


            // draw the drift optimization circle
            //circle(frame.frame, this->lastKnownPosition, (int)driftRadius, colorDrift, 1);
			//circle(frame.frame, this->lastKnownPosition, 5, colorDrift, 2);
        }

    } 
    else{     // perform OBJECT TRACKING
        Rect2d bbox;
        bool located = tracker->update(frame.frame, bbox);
        if(located){
            // Bounding box in tracking located
            leaderPosition = bbox; // centerBB(bbox); //add information for return
            if(showLog){
                rectangle(frame.frame, bbox, colorTracking, 2);
                //circle(frame.frame, leaderPosition, 5, colorTracking, 2);
            }
        }
    }
    
    //>>>>> show frame to video
    if(showLog){
        bool quit = showFrame(frame.frame);
        if(quit){
            leaderPosition = breakPosition;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////   Additional Function section   ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

void Follower3D::addNegativeToKNN(){
    // load an image from the dataset of negative people
    Mat negativePerson;
    loadNegativePerson(negativePerson);
    // generate the encodings for it and train KNN
    vector<float> encoding;
    classifier->feed(negativePerson, encoding);
    knn->trainWithOne(encoding, SOMEONE);
}

void Follower3D::addBBsToKNN(const Mat & frame, const vector<vector<int> > & detections, int leaderIndex){
    for(int i=0; i<detections.size(); i++){

        // crop the ROI and generate its encoding to train KNN (according to the labels)
        Mat cropBB = frame( Rect(detections[i][2], detections[i][3], detections[i][4], detections[i][5]) );
        vector<float> encoding;
        classifier->feed(cropBB, encoding);

        int label = ( (i==leaderIndex) ? LEADER : SOMEONE );
        knn->trainWithOne(encoding, label);
    }

    if(leaderIndex>=0 && detections.size()==1){
        // AKA I have only the leader, then I choose a negative to improve the KNN space
        addNegativeToKNN();
    }
}

void Follower3D::drawDetectionsToFrame(Mat & frame, const vector<vector<int> > & detections, const Scalar & colorSomeone, int leaderIndex, const Scalar & colorLeader, bool drawCenter){
    for(int i=0; i<detections.size(); i++){
        // the color is chosen according to the labels
        Scalar color = ( (i==leaderIndex) ? colorLeader : colorSomeone );
        detector->drawOnePrediction(detections[i], frame, color, true);
        // each BB is precisely located with a small circle on the centre
        if(drawCenter){
            circle(frame, centerBB(detections[i][2], detections[i][3], detections[i][4], detections[i][5]), 5, color, 2);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////   Utility Function section   //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
void Follower3D::loadNegativePerson(Mat & image){
    // use a static variable so next call will start from the current position
    static int nextPerson = randUniform(0, 5180); //the dataset contain exactly 5180 images

    // build the path and load the image
    string folder = negativePeopleDataset + "set" + std::to_string(nextPerson/518) + "/";
    string imageName = std::to_string(nextPerson%518) + ".jpg";
    string path = folder + imageName;
    image = cv::imread(path);

    // update the counter and then apply the module to respect the dataset bounds
    nextPerson = (++nextPerson)%5180;
}


Point2i Follower3D::centerBB(int left, int top, int width, int height){
    // calculate the center of a BB of int
    int centerX = left + (width /2);
    int centerY = top + (height /2);
    return(Point2i(centerX, centerY));
}

Point2i Follower3D::centerBB(const Rect2d & rect){
    // calculate the center of a BB of double
    return centerBB((int)rect.x, (int)rect.y, (int)rect.width, (int)rect.height);
}

bool Follower3D::showFrame(Mat & frame){
    // Display FPS on frame
    string fpsStr = doubleToString(videoIn->fps(), 2);
    cv::putText(frame, "FPS : " + fpsStr, Point2i(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255,255,255), 2);

    // downscale and show the image
    downScaleImageTo(frame, downscaleWidth);
    return videoOut->addFrame(frame);
}



bool Follower3D::bbox2World(const FrameData3D & frame, const Rect2d & bbox, float & xref, float & yref) {
    bool valid = false;
    double dmin = 1e9;
    xref = 0;
    yref = 0;

    Point2i center = {
        bbox.x+bbox.width/2,
        bbox.y+bbox.height/2 
    };

    int dl = (bbox.height/2)/N_VERT_SAMPLES;
    for (int i=0; i<N_VERT_SAMPLES; ++i) {
        int yi = center.y - i*dl;  
        float x, y, z;
        if (videoIn->worldCoordinates(frame, center.x, yi, x, y, z)) {
            //std::cout << x << " " << y << " " << z << std::endl;
            //hist.push_back({x, z});
            if (z < dmin) {
                valid = true;
                dmin = z;
                xref = x;
                yref = z;
            }
        }
    }
    return valid;
}
