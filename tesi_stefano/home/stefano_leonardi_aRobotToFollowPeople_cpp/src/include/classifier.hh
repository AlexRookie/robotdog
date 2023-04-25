#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

class Classifier {
public:
    virtual ~Classifier() {};

    virtual void feed(const cv::Mat & image, std::vector<float> & output) = 0;
};