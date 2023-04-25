#pragma once

// #include "dnnModel.hh"
// #include "follower3D.hh"
// #include "utils.hh"
// #include "videoIO3D.hh"

#include <jsoncons/json.hpp>
#include <jsoncons_ext/cbor/cbor.hpp>
#include <jsoncons_ext/cbor/cbor_cursor.hpp>
#include <jsoncons_ext/jsonpath/jsonpath.hpp>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <memory>

//using namespace jsoncons; // for convenience

class Player3D {
public:
    Player3D(std::string filename);
    ~Player3D();
    void play();

private:
    std::string filename;

    cv::Mat PNG2image(const std::vector<uint8_t>& buffer);    

};