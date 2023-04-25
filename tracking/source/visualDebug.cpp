//============================================================================/
// Visual Debug
//
// Alessandro Antonucci @AlexRookie
// Paolo Bevilacqua
// University of Trento
//============================================================================/

#include <stdio.h>
#include <stdlib.h> // exit(0);
#include <iostream>
#include <fstream>

#include <cmath>
#include <vector>

#include <chrono>
#include <ctime>
#include <string>

#include <signal.h>
#include <unistd.h>
#include <getopt.h>

#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp> // OpenCV
#include "zmq/Publisher.hh"
#include "zmq/Subscriber.hh"
#include "json.hpp"
#include "params.hpp"

#include "MapManager.hpp"
#include "utils.hpp"
#include "ClothoidList.hh"

using nlohmann::json;

static inline void delay(int ms) {
    while (ms>=1000){
        usleep(1000*1000);
        ms-=1000;
    };
    if (ms!=0)
        usleep(ms*1000);
}

bool ctrl_c_pressed;
void ctrlc(int) {
    ctrl_c_pressed = true;
}

long long int timeNow() {
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    using std::chrono::system_clock;
    return (long long int)duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

struct LocData {
    double x = 0, y = 0, theta = 0;
    long long int timestamp = 0;
    std::mutex mtx;
};

struct CamData {
    double x = 0, y = 0;
    long long int timestamp = 0;
    bool valid = false;
    std::mutex mtx;
};

struct LdrData {
    std::vector<double> xs, ys;
    std::vector<int> ids;
    std::vector<bool> valids;
    long long int timestamp = 0;
    std::mutex mtx;
};

struct FusData {
    double x = 0, y = 0;
    long long int timestamp = 0;
    bool valid = false;
    std::mutex mtx;
};

struct FilData {
    double meas_x_rel = 0, meas_y_rel = 0;
    double meas_x_abs = 0, meas_y_abs = 0;
    bool meas_valid = false;
    double kf_x = 0, kf_y = 0, ekf_x = 0, ekf_y = 0;
    long long int timestamp = 0;
    std::mutex mtx;
};

struct PathData {
    G2lib::ClothoidList path;
    long long int timestamp = 0;
    std::mutex mtx;
};

// PATH DESERIALIZATION FOR DEBUGGING
namespace G2lib {

    void from_json(const json& j, ClothoidList& cl) {
        for (int i=0; i<j.size(); ++i) {
            G2lib::ClothoidCurve cc = j[i].get<G2lib::ClothoidCurve>();
            cl.push_back(cc);
        }
    }

    void from_json(const json& j, ClothoidCurve& cc) {
        double x0 = j.at("x0");
        double y0 = j.at("y0");
        double t0 = j.at("t0");
        double k0 = j.at("k0");
        double dk = j.at("dk");
        double L  = j.at("L");
        cc.build(x0, y0, t0, k0, dk, L);
    }

};


int main(int argc, char ** argv) {
	signal(SIGINT, ctrlc);

    if (argc<2) {
        std::cout << "Usage: " << argv[0] << " <paramsFile>" << std::endl;
        return 0;
    }
    ParamsProvider::init(argv[1]);

    std::string subAddressLoc     = ParamsProvider::getInstance().getSubFullAddress("loc");
    std::string subAddress3D      = ParamsProvider::getInstance().getSubFullAddress("3Dcam");
    std::string subAddressCluster = ParamsProvider::getInstance().getSubFullAddress("cluster");
    std::string subAddressFusion  = ParamsProvider::getInstance().getSubFullAddress("humPos");
    std::string subAddressFilter  = "tcp://127.0.0.1:1234"; //ParamsProvider::getInstance().getSubFullAddress("");
    std::string subAddressPath    = ParamsProvider::getInstance().getSubFullAddress("path");

    std::string topicLoc     = ParamsProvider::getInstance().getTopic("loc");
    std::string topic3D      = ParamsProvider::getInstance().getTopic("3Dcam");
    std::string topicCluster = ParamsProvider::getInstance().getTopic("cluster");
    std::string topicFusion  = ParamsProvider::getInstance().getTopic("humPos");
    std::string topicFilter  = "FIL_POS"; 
    std::string topicPath    = ParamsProvider::getInstance().getTopic("path");

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO LOC
    ////////////////////////////////////////////////////////////////////////////

    Subscriber subLoc;
    LocData locData;
    subLoc.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        try {
            json jobj = json::parse(std::string(buf, size));
            long long int timestamp = timeNow();
            double x = jobj.at("loc_data").at("x"); 
            double y = jobj.at("loc_data").at("y");
            double theta = jobj.at("loc_data").at("theta");

            {
                std::unique_lock<std::mutex> lck(locData.mtx);
                locData.x = x; 
                locData.y = y;
                locData.theta = theta;
                locData.timestamp = timestamp;	    
            }
        }
        catch(std::exception &e) {
            std::cerr << "Error parsing loc data: " << e.what() << std::endl;
        }
    });
    subLoc.start(subAddressLoc, topicLoc);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO CAMERA
    ////////////////////////////////////////////////////////////////////////////

    Subscriber sub3D;
    CamData track3D;
    sub3D.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        try {
            json jobj = json::parse(std::string(buf, size));
            long long int timestamp = jobj.at("ts");
            double x = jobj.at("x"); 
            double y = jobj.at("y");
            bool valid = jobj.at("valid");

            {
                std::unique_lock<std::mutex> lck(track3D.mtx);
                track3D.x = x; 
                track3D.y = y;
                track3D.valid = valid;
                track3D.timestamp = timestamp;	    
            }
        }
        catch(std::exception &e) {
            std::cerr << "Error parsing 3D camera data: " << e.what() << std::endl;
        }
    });
    sub3D.start(subAddress3D, topic3D);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO CLUSTERS
    ////////////////////////////////////////////////////////////////////////////

    Subscriber subCluster;
    LdrData trackLdr;
    subCluster.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        try {
            json jobj = json::parse(std::string(buf, size));
            long long timestamp = jobj.at("ts");
            json jclusters = jobj.at("clusters");
            size_t sz = jclusters.size();
            std::vector<double> xs, ys;
            std::vector<int> ids;
            std::vector<bool> valids;
            xs.reserve(sz);
            ys.reserve(sz);
            ids.reserve(sz);
            valids.reserve(sz);
            
            for (int i=0; i<sz; ++i) {
                double x = jclusters[i].at("x");
                double y = jclusters[i].at("y");
                int id = jclusters[i].at("id");
                bool valid = jclusters[i].at("visible");
                xs.push_back(x);
                ys.push_back(y);
                ids.push_back(id);
                valids.push_back(valid);
            }
            {
                std::unique_lock<std::mutex> lck(trackLdr.mtx);
                trackLdr.xs = xs;
                trackLdr.ys = ys;
                trackLdr.ids = ids;
                trackLdr.valids = valids;
                trackLdr.timestamp = timestamp;
            }
        }
        catch(std::exception &e) {
            std::cerr << "Error parsing lidar data: " << e.what() << std::endl;
        }
    });
    subCluster.start(subAddressCluster, topicCluster);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO FUSION
    ////////////////////////////////////////////////////////////////////////////

    Subscriber subFusion;
    FusData trackFusion;
    subFusion.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        try {
            json jobj = json::parse(std::string(buf, size));
            long long int timestamp = jobj.at("ts");
            double x = jobj.at("x"); 
            double y = jobj.at("y");
            bool valid = jobj.at("valid");
	    
            {
                std::unique_lock<std::mutex> lck(trackFusion.mtx);
                trackFusion.x = x; 
                trackFusion.y = y;
                trackFusion.timestamp = timestamp;	    
                trackFusion.valid = valid;	    
            }
        }
        catch(std::exception &e) {
            std::cerr << "Error parsing lidar data: " << e.what() << std::endl;
        }
    });
    subFusion.start(subAddressFusion, topicFusion);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO FILTER
    ////////////////////////////////////////////////////////////////////////////

    Subscriber subFilter;
    FilData trackFilter;
    //std::string logname = "logFilter_" + std::to_string(timeNow()) + ".txt";
    //std::ofstream logfile(logname);
    subFilter.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        try {
            std::string msg = std::string(buf, size);
            json jobj = json::parse(msg);
            long long int timestamp = jobj.at("ts");
            double meas_x_rel = jobj.at("meas_x_rel"); 
            double meas_y_rel = jobj.at("meas_y_rel");
            double meas_x_abs = jobj.at("meas_x_abs");
            double meas_y_abs = jobj.at("meas_y_abs");
            bool   meas_valid = jobj.at("meas_valid");
            double kf_x       = jobj.at("kf_x");
            double kf_y       = jobj.at("kf_y");
            double ekf_x      = jobj.at("ekf_x");
            double ekf_y      = jobj.at("ekf_y");
            double loc_x      = jobj.at("loc_x");
            double loc_y      = jobj.at("loc_y");
            double loc_theta  = jobj.at("loc_theta");
            
            {
                std::unique_lock<std::mutex> lck(trackFilter.mtx);
                trackFilter.timestamp  = timestamp  ;
                trackFilter.meas_x_rel = meas_x_rel ;
                trackFilter.meas_y_rel = meas_y_rel ;
                trackFilter.meas_x_abs = meas_x_abs ;
                trackFilter.meas_y_abs = meas_y_abs ;
                trackFilter.meas_valid = meas_valid ;
                trackFilter.kf_x       = kf_x       ;
                trackFilter.kf_y       = kf_y       ;
                trackFilter.ekf_x      = ekf_x      ;
                trackFilter.ekf_y      = ekf_y      ;
            }

            //logfile << msg << std::endl;
        }
        catch(std::exception &e) {
            std::cerr << "Error parsing lidar data: " << e.what() << std::endl;
        }
    });
    subFilter.start(subAddressFilter, topicFilter);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO PATH
    ////////////////////////////////////////////////////////////////////////////
    
    Subscriber subPath;
    PathData pathData;
    subPath.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        try {
            json jobj = json::parse(std::string(buf, size));
            G2lib::ClothoidList cl = jobj["path"].get<G2lib::ClothoidList>();
            {
                std::unique_lock<std::mutex> lck(pathData.mtx);
                pathData.path = cl; 
                pathData.timestamp = timeNow();	    
            }
        }
        catch(std::exception &e) {
            std::cerr << "Error parsing path data: " << e.what() << std::endl;
        }
    });
    subPath.start(subAddressPath, topicPath);

    std::string mapname = "data/mappaRobodogC.yaml";
    PosedMap map;
    if (!loadMapFile(mapname, map)) {
        throw std::runtime_error("Failed to load map \"" + mapname + "\"");
    }
    MapManager mm(200, 0.05, map);

    std::vector<cv::Point3f> colors = {
        {1.0, 0.0, 0.0},
        {0.5, 0.5, 0.5},
        {0.8, 0.2, 0.2},
        {0.1, 0.5, 1.0},
        {0.7, 0.7, 0.7},
        {0.1, 0.5, 0.1},
        {0.8, 0.7, 0.5},
        {0.8, 0.6, 0.1},
        {0.2, 0.2, 0.2},
        {0.0, 0.0, 0.5}
    };

    ////////////////////////////////////////////////////////////////////////////
    // PROCESSING
    ////////////////////////////////////////////////////////////////////////////
    
    std::cout << "[Looping...]" << std::endl;
    while (1) {

        ////////////////////////////////////////////////////////////////////////
        // RETRIEVE DATA
        ////////////////////////////////////////////////////////////////////////
        LocData locDataC;
        {
            std::unique_lock<std::mutex> lck(locData.mtx);
            locDataC.x = locData.x;
            locDataC.y = locData.y;
            locDataC.theta = locData.theta;
            locDataC.timestamp = locData.timestamp;
        }
        
        CamData track3DC;
        {
            std::unique_lock<std::mutex> lck(track3D.mtx);
            track3DC.x = track3D.x;
            track3DC.y = track3D.y;
            track3DC.valid = track3D.valid;
            track3DC.timestamp = track3D.timestamp;
        }

        LdrData trackLdrC;
        {
            std::unique_lock<std::mutex> lck(trackLdr.mtx);
            trackLdrC.xs = trackLdr.xs;
            trackLdrC.ys = trackLdr.ys;
            trackLdrC.ids = trackLdr.ids;
            trackLdrC.valids = trackLdr.valids;
            trackLdrC.timestamp = trackLdr.timestamp;            
        }

        FusData trackFusC;
        {
            std::unique_lock<std::mutex> lck(trackFusion.mtx);
            trackFusC.x = trackFusion.x;
            trackFusC.y = trackFusion.y;
            trackFusC.timestamp = trackFusion.timestamp;            
            trackFusC.valid = trackFusion.valid;            
        }

        FilData trackFilC;
        {
            std::unique_lock<std::mutex> lck(trackFilter.mtx);
            trackFilC.timestamp  = trackFilter.timestamp   ;
            trackFilC.meas_x_rel = trackFilter.meas_x_rel  ;
            trackFilC.meas_y_rel = trackFilter.meas_y_rel  ;
            trackFilC.meas_x_abs = trackFilter.meas_x_abs  ;
            trackFilC.meas_y_abs = trackFilter.meas_y_abs  ;
            trackFilC.meas_valid = trackFilter.meas_valid  ;
            trackFilC.kf_x       = trackFilter.kf_x        ;
            trackFilC.kf_y       = trackFilter.kf_y        ;
            trackFilC.ekf_x      = trackFilter.ekf_x       ;
            trackFilC.ekf_y      = trackFilter.ekf_y       ;
        }

        PathData pathDataC;
        {
            std::unique_lock<std::mutex> lck(pathData.mtx);
            pathDataC.path = pathData.path;
            pathDataC.timestamp = pathData.timestamp;            
        }
        
        long long int currTs = timeNow();
        
        ///////////////////////////////////////////////////////////////////////
        // PLOT RELATIVE MAP
        ///////////////////////////////////////////////////////////////////////

        // Plot map grid
        int l = 800;
        cv::Mat plot(l ,l, CV_8UC3, cv::Scalar(255,255,255));
        for (int i=50; i<=l; i+=50) {
            cv::line(plot, cv::Point(i, 0), cv::Point(i, l), cv::Scalar(128,128,128), 1);
            cv::line(plot, cv::Point(0, i), cv::Point(l, i), cv::Scalar(128,128,128), 1);
        }
        
        // Plot axis
        int px = l/2;
        int py = l/2;
        cv::arrowedLine(plot, cv::Point2f(px,py), cv::Point2f(px+l*49./100,py), cv::Scalar(0,0,0), 2, cv::LINE_AA, 0, 0.03);
        cv::arrowedLine(plot, cv::Point2f(px,py), cv::Point2f(px,py+l*49./100), cv::Scalar(0,0,0), 2, cv::LINE_AA, 0, 0.03);


        //track3DC.x -= 2*0.28;
        // Plot camera object
        if (track3DC.valid) {
            px = l/2 + (int)(track3DC.x*50);
            py = l/2 + (int)(track3DC.y*50);
            cv::rectangle(plot, cv::Point2f(px,py), cv::Point2f(px+7,py+7), cv::Scalar(255,0,0), -1);
        }

        // Plot clusters
        for (int i=0; i < trackLdrC.xs.size(); i++) {
            px = l/2 + (int)(trackLdrC.xs[i]*50);
            py = l/2 + (int)(trackLdrC.ys[i]*50);
            int color_id = trackLdrC.ids[i] % colors.size();
            cv::rectangle(plot, cv::Point2f(px,py), cv::Point2f(px+5,py+5), cv::Scalar(colors[color_id].z*255, colors[color_id].y*255, colors[color_id].x*255), -1);
            cv::putText(plot, std::to_string(trackLdrC.ids[i]), cv::Point2f(px,py+10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1.2, cv::LINE_AA, true);
        }

        // Plot fusion
        if (trackFusC.valid) {
            px = l/2 + (int)(trackFusC.x*50);
            py = l/2 + (int)(trackFusC.y*50);
            cv::rectangle(plot, cv::Point2f(px,py), cv::Point2f(px+10,py+10), cv::Scalar(0,255,0), -1);
        }

        cv::flip(plot, plot, 0);
        cv::imshow("TrackingRel", plot);

        ///////////////////////////////////////////////////////////////////////
        // PLOT ABSOLUTE MAP
        ///////////////////////////////////////////////////////////////////////
        
        // Plot map
        mm.setOrigin(locDataC.x, locDataC.y);
        cv::Mat plotAbs = mm.getMapWindow(); 

        // Plot loc
        int r1, c1;
        mm.getMapCoordinatesAbs(locDataC.x, locDataC.y, r1, c1);
        int r2, c2;
        mm.getMapCoordinatesAbs(locDataC.x + std::cos(locDataC.theta), locDataC.y + std::sin(locDataC.theta), r2, c2);
        cv::arrowedLine(plotAbs, cv::Point2f(c1, r1), cv::Point2f(c2, r2), cv::Scalar(255,0,0), 2, cv::LINE_AA, 0, 0.3);
        
        // Plot filter
        int r, c;
        mm.getMapCoordinatesAbs(trackFilC.kf_x, trackFilC.kf_y, r, c);
        cv::rectangle(plotAbs, cv::Point2f(c, r), cv::Point2f(c+5, r+5), cv::Scalar(0,80,0), -1);
        mm.getMapCoordinatesAbs(trackFilC.ekf_x, trackFilC.ekf_y, r, c);
        cv::rectangle(plotAbs, cv::Point2f(c, r), cv::Point2f(c+5, r+5), cv::Scalar(0,0,80), -1);        
    
        // Plot path
        const G2lib::ClothoidList& path = pathDataC.path;

        if (path.numSegment()>0) {
            int ncuts = std::ceil(path.length()/0.1);
            double dL = path.length()/ncuts;

            double x0 = path.xBegin(), y0 = path.yBegin();
            for (int i=1; i<=ncuts; ++i) {
                double s0 = i*dL;
                double xs = path.X(s0);
                double ys = path.Y(s0);
                int r0, c0, rs, cs;
                mm.getMapCoordinatesAbs(x0, y0, r0, c0);
                mm.getMapCoordinatesAbs(xs, ys, rs, cs);
                cv::line(plotAbs, cv::Point2f(c0, r0), cv::Point2f(cs, rs), cv::Scalar(0,180,0), 1, cv::LINE_AA);
                x0 = xs;
                y0 = ys;
            }
        }

        cv::flip(plotAbs, plotAbs, 0);
        cv::resize(plotAbs, plotAbs, cv::Size(plotAbs.cols*2, plotAbs.rows*2), cv::INTER_CUBIC);
        cv::imshow("TrackingAbs", plotAbs);
        
        int key = cv::waitKey(50);
    
        // Exit
        if (ctrl_c_pressed) {
            break;
        }
    }

    subLoc.stop();
    sub3D.stop();
    subCluster.stop();
    subFusion.stop();
    subFilter.stop();
    subPath.stop();

    //logfile.close();

    return 0;
}
