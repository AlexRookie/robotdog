#include "logger3D.hh"


bool Logger3D::bbox2World(const FrameData3D & frame, const cv::Rect2d & bbox, float & xref, float & yref) {
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


Logger3D::Logger3D(DetectorNames detectorName, std::string filename, float confThresh): 
    confThresh(confThresh)
{
    // Init input video
    videoIn = std::make_unique<LoadVideo3D>(1000);

    switch(detectorName){
        case SSD:     detector = std::make_unique<MobileNetSSD>(); break;
        case YOLO:    detector = std::make_unique<YOLOv3>();       break;
        default:      cout << "[INFO] Warning: no detector were chosen...\n";
    }
    detector->useCuda();

    os = std::make_unique<std::ofstream>(filename);
    if (!os->is_open()) {
        throw std::runtime_error("Failed to open file \"" + filename + "\"");
    }

    encoder = std::make_unique<jsoncons::cbor::cbor_stream_encoder>(*os);
    encoder->begin_array();
}   

Logger3D::~Logger3D() {
    encoder->end_array();
    encoder->flush();
    encoder.reset();
    os->close();
    os.reset();
    detector.reset();
}

void Logger3D::log(){
    FrameData3D frame = videoIn->read();

    if(frame.frame.empty()){
        cout << "[INFO] End of video source\n";
    } else{

        //>>>>> look for detections in the image
        std::vector<vector<int> > detections;
        detector->detectPeopleOnly(frame.frame, detections, confThresh);

        size_t sz = detections.size();
        std::vector<std::vector<int>> rectBB(sz);
        std::vector<std::vector<float>> cw(sz, std::vector<float>(2));
        std::vector<uint8_t> valid(sz);
        for(size_t i=0; i<sz; i++){
            // get bounding box of i^th detection 
            rectBB[i] = { detections[i][2], detections[i][3], detections[i][4], detections[i][5] };
            
            // get camera coordinates of i^th detection 
            valid[i] = bbox2World(frame, Rect(detections[i][2], detections[i][3], detections[i][4], detections[i][5]), cw[i][0], cw[i][1]);
        }

        encoder->begin_object();
        encoder->key("t");
        encoder->int64_value(frame.lastTime);
        encoder->key("f");
        auto bytes = image2PNG(frame.frame);
        encoder->typed_array(jsoncons::span<uint8_t>(bytes.data(), bytes.size()));
        encoder->key("b");
        encoder->begin_array();
        for (int i=0; i<sz; i++) {
            encoder->typed_array(jsoncons::span<int>(rectBB[i].data(), rectBB[i].size()));
        }
        encoder->end_array();
        encoder->key("w");
        encoder->begin_array();
        for (int i=0; i<sz; i++) {
            encoder->typed_array(jsoncons::span<float>(cw[i].data(), cw[i].size()));
        }
        encoder->end_array();
        encoder->key("v");
        encoder->typed_array(jsoncons::span<uint8_t>(valid.data(), valid.size()));
        encoder->end_object();

        cv::imshow("win", frame.frame);
        cv::waitKey(1);
    }

}

std::vector<uint8_t> Logger3D::image2PNG(const cv::Mat& image)
{
    if (image.empty() || (image.cols == 0 && image.rows == 0))
        return {};

    
    std::vector<uint8_t> buffer;

    // Set the image compression parameters.
    std::vector<int> params;
    // if (extension == ".png")
    // {
    // params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    // params.push_back(6);     // MAX_MEM_LEVEL = 9
    // params.push_back(cv::IMWRITE_PNG_STRATEGY);
    // params.push_back(cv::IMWRITE_PNG_STRATEGY_RLE);
    // if (!cv::imencode(".png", image, buffer, params)) {
    //     throw std::runtime_error("FAILED TO COMPRESS IMAGE TO PNG");
    // }
    // }
    // else if (extension == ".jpg")
    // {
    //     params.push_back(cv::IMWRITE_JPEG_QUALITY);
    //     params.push_back(90);     // 0...100 (higher is better)
    //     success = cv::imencode(extension, image, buffer, params);
    // }

    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(95);
    if (!cv::imencode(".jpg", image, buffer, params)) {
        throw std::runtime_error("FAILED TO COMPRESS IMAGE TO JPG");
    }

    return buffer;
}
