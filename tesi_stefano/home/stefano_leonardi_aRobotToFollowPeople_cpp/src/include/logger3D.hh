#pragma once

#include "dnnModel.hh"
#include "follower3D.hh"
#include "utils.hh"
#include "videoIO3D.hh"

#include <jsoncons/json.hpp>
#include <jsoncons_ext/cbor/cbor.hpp>
#include <jsoncons_ext/cbor/cbor_cursor.hpp>
#include <jsoncons_ext/jsonpath/jsonpath.hpp>

#include <memory>

//using namespace jsoncons; // for convenience

class Logger3D {
public:
    Logger3D(DetectorNames detectorName, std::string filename, float confThresh);
    ~Logger3D();
    void log();

private:
    std::unique_ptr<LoadVideo3D> videoIn;
    std::unique_ptr<DetectorModel> detector;
    std::unique_ptr<std::ofstream> os;
    std::unique_ptr<jsoncons::cbor::cbor_stream_encoder> encoder;

    float confThresh;

    std::vector<uint8_t> image2PNG(const cv::Mat& image);
    bool bbox2World(const FrameData3D & frame, const cv::Rect2d & bbox, float & xref, float & yref);

};