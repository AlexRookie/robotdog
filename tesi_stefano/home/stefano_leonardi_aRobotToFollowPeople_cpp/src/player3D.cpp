#include "player3D.hh"



Player3D::Player3D(std::string filename): 
    filename(filename)
{}   

Player3D::~Player3D() 
{}

void Player3D::play(){
    std::ifstream is(filename);    
    if (!is.is_open()) {
        throw std::runtime_error("Failed to open file \"" + filename + "\"");
    }
    
    jsoncons::ojson j2 = jsoncons::cbor::decode_cbor<jsoncons::ojson>(is);
    std::cout << j2.size() << std::endl;

    int64_t t0 = 0;

    for (int i=0; i<j2.size(); ++i) {
        jsoncons::ojson obj = j2[i].as<jsoncons::ojson>();
        
        int64_t timestamp = obj["t"].as<int64_t>();
        
        if (t0>0) {
            int delay = timestamp-t0;
            //std::cerr << delay << std::endl;
            cv::waitKey(delay);
        }
        t0 = timestamp;
        std::cerr << t0 << std::endl;

        std::vector<uint8_t> bytes = obj["f"].as<std::vector<uint8_t>>();
        auto img = PNG2image(bytes);

        jsoncons::ojson bboxes = obj["b"].as<jsoncons::ojson>();
        //std::cout << bboxes.size() << std::endl;
        for (int j=0; j<bboxes.size(); ++j) {
            // LEFT=2, TOP=3, WIDTH=4, HEIGHT=5
            int left = bboxes[j][0].as<int>();
            int top = bboxes[j][1].as<int>();
            int right  = (left + bboxes[j][2].as<int>());
            int bottom = (top + bboxes[j][3].as<int>());
            cv::Scalar color = cv::Scalar(255, 0, 0);
            cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom), color, 2);
        }
        
        cv::imshow("win", img);
    }

}

cv::Mat Player3D::PNG2image(const std::vector<uint8_t>& buffer)
{
    return cv::imdecode(buffer, cv::IMREAD_COLOR);
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " logfile" << std::endl;
        return 0;
    }

    Player3D player(argv[1]);
    player.play();
}