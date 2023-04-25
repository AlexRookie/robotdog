//============================================================================/
// Main fusion
//
// Alessandro Antonucci @AlexRookie
// Paolo Bevilacqua
// University of Trento
//============================================================================/

#include <stdio.h>
#include <stdlib.h> // exit(0);
#include <iostream>

#include <cmath>
#include <vector>
#include <limits>

#include <chrono>
#include <ctime>
#include <string>

#include <signal.h>
#include <unistd.h>
#include <getopt.h>

#include <thread>
#include <mutex>

#include "zmq/Publisher.hh"
#include "zmq/Subscriber.hh"
#include "json.hpp"

#include "munkres.h"
#include "std2dvectordapter.h"

#include "params.hpp"

using nlohmann::json;

// LIDAR PARAMETERS
#define VECTOR_SIZE (360*2)
#define OFFSET_DEGS -90.0
#define CLUST_TOL   0.3   // clustering distance tolerance [m]
#define MIN_POINTS  10    // min number of points per cluster
#define EPSILON     0.2   // Ramer-Douglas–Peucker threshold distance 

// CLUSTER ASSOCIATION PARAMETERS
#define NO_ASSOCIATION              std::numeric_limits<int>::max()
#define COST_UNASSOCIATED_TRACK     40 // [cm]
#define COST_UNASSOCIATED_DETECTION 40 // [cm]
#define FAIL_DETECTION              30

// OLD STATE MACHINE PARAMETERS
// #define TIME_TOL     0.7  // [m]
// #define MAX_FAILURES 5

// STATE MACHINE PARAMETERS
#define D_THRESH          0.7  // [m]
#define VALID_TIME        500. // [ms]
#define DIST_PROMOTION    0.7  // [m]
#define LEADER_PROMOTION  3
#define LEADER_REJECTION  10

#define ASSERT(cond) \
    do \
    { \
        if (!(cond)) \
        { \
            std::cerr << "File: " << __FILE__ << " Line: " << __LINE__ << "Assertion failed: " << #cond << std::endl; \
            exit(0); \
        } \
    } while(0)  


template <typename T> 
static inline int sgn(T val){
    return (T(0)<val) - (val < T(0));
}

struct Point {
    double x;
    double y;

    friend Point operator-(Point const & lhs, Point const & rhs) {
        return {lhs.x-rhs.x, lhs.y-rhs.y};
    }

    friend Point operator/(Point const & lhs, double rhs) {
        return {lhs.x/rhs, lhs.y/rhs};
    }
};

struct cluster {
    Point pos;
    Point vel;
    std::vector<Point> points;
    std::vector<Point> filt_points;
    //Point angles;
    int id;
    int cnt_no_det = 0;
    int cnt_prom = 0;
    enum CType {LEADER, OBSTACLE, UNKNOWN};
    CType ctype;
    long long int timestamp = 0;
};

static inline double diffangle(double beta, double alpha){
    // set alpha, beta between 0 and 2*pi
    while(fabs(beta)>=2.0*M_PI){
        beta = beta - 2.0*M_PI*sgn(beta);
    }
    if(beta<0.0){
        beta = beta + 2.0*M_PI;
    }
    while(fabs(alpha)>2.0*M_PI){
        alpha = alpha - 2.0*M_PI*sgn(alpha);
    }
    if(alpha<0.0){
        alpha = alpha + 2.0*M_PI;
    }
    double difference = beta - alpha;
    if(difference>M_PI){
        difference = difference - 2.0*M_PI;
    }
    if(difference<-M_PI){
        difference = difference + 2.0*M_PI;
    }
    return difference;  
}

struct EulerAngles {
  double roll, pitch, yaw;
};

struct Quaternion {
  double x, y, z, w;
};

EulerAngles ToEulerAngles(Quaternion const & q);

// Perpendicular distance point to segment
double perpendicularDistance(const Point &pt, const Point &lineStart, const Point &lineEnd) {
    double dx = lineEnd.x - lineStart.x;
    double dy = lineEnd.y - lineStart.y;
 
    // Normalise
    double mag = std::pow(std::pow(dx,2.0)+std::pow(dy,2.0),0.5);
    if(mag > 0.0) {
        dx /= mag; dy /= mag;
    }
 
    double pvx = pt.x - lineStart.x;
    double pvy = pt.y - lineStart.y;
 
    // Get dot product (project pv onto normalized direction)
    double pvdot = dx * pvx + dy * pvy;
 
    //Scale line direction vector
    double dsx = pvdot * dx;
    double dsy = pvdot * dy;
 
    //Subtract this from pv
    double ax = pvx - dsx;
    double ay = pvy - dsy;
 
    return std::pow(std::pow(ax,2.0)+std::pow(ay,2.0),0.5);
}

// Ramer-Douglas–Peucker algorithm for reducing the number of points in a curve that is approximated by a series of points
void RamerDouglasPeucker(const std::vector<Point> &pointList, double epsilon, std::vector<Point> &out) {
    if (pointList.size()<2) {
        out = pointList;
        return;
    }

    // Find the point with the maximum distance from line between start and end
    double dmax = 0.0;
    size_t index = 0;
    size_t end = pointList.size()-1;
    for (size_t i = 1; i < end; i++) {
        double d = perpendicularDistance(pointList[i], pointList[0], pointList[end]);
        if (d > dmax) {
            index = i;
            dmax = d;
        }
    }
 
    // If max distance is greater than epsilon, recursively simplify
    if (dmax > epsilon) {
        // Recursive call
        std::vector<Point> recResults1;
        std::vector<Point> recResults2;
        std::vector<Point> firstLine(pointList.begin(), pointList.begin()+index+1);
        std::vector<Point> lastLine(pointList.begin()+index, pointList.end());
        RamerDouglasPeucker(firstLine, epsilon, recResults1);
        RamerDouglasPeucker(lastLine, epsilon, recResults2);
 
        // Build the result list
        out.assign(recResults1.begin(), recResults1.end()-1);
        out.insert(out.end(), recResults2.begin(), recResults2.end());
        ASSERT(out.size()>=2);
    } 
    else {
        // Just return start and end points
        out.clear();
        out.push_back(pointList[0]);
        out.push_back(pointList[end]);
    }
}

// Polar to cartesian coordinates
void polar2cartesian(double *scan_dist, double *scan_angle, double *cart_x, double *cart_y, const int &size) {
    // Flush data array
    for (int i=0; i < size; i++) {
        cart_x[i] = 0.0;
        cart_y[i] = 0.0;
    }

    for (int i=0; i < size; i++) {
        if (scan_dist[i] == 0.0) {
            continue;
        }
        cart_x[i] = scan_dist[i] * cos((scan_angle[i]+90.0)*M_PI/180.0);
        cart_y[i] = scan_dist[i] * sin((scan_angle[i]+90.0)*M_PI/180.0);
    }
}

// Lidar points clustering
void clusterData(double *cart_x, double *cart_y, const int &size, std::vector<Point> &centroids, std::vector<std::vector<Point>> &points, std::vector<std::vector<Point>> &filt_points) {
    std::vector<Point> clust_points;
    //std::vector<double> x;
    //std::vector<double> y;
    centroids.clear();
    points.clear();
    filt_points.clear();

    for (int i=0; i<size; i++) {
        if (cart_x[i] == 0.0 && cart_y[i] == 0.0) continue;

        if (clust_points.empty()) {
            // New object
            clust_points.push_back(Point({cart_x[i],cart_y[i]}));
            //x.push_back(cart_x[i]);
            //y.push_back(cart_y[i]);
            //angle1 = scan_angle[i]+90.0;
        } else {
            double dist = std::hypot(clust_points.back().x - cart_x[i], clust_points.back().y - cart_y[i]);
            //double dist = std::sqrt( std::pow(clust_points.back().x - cart_x[i], 2) + std::pow(clust_points.back().y - cart_y[i], 2) );
            if (dist > CLUST_TOL) {
                if (clust_points.size() >= MIN_POINTS) {
                    // Reduce number of points
                    std::vector<Point> reduced_points;
                    //for (int ii=0; ii<x.size(); ii++) {
                    //    all_points.push_back(Point({x[ii],y[ii]}));
                    //}
                    RamerDouglasPeucker(clust_points, EPSILON, reduced_points);
                    points.push_back(clust_points);
                    filt_points.push_back(reduced_points);
                    // Cluster centroid
                    double sum_x = 0., sum_y = 0.;
                    for (int ii=0; ii<clust_points.size(); ii++) {
                        sum_x += clust_points[ii].x; 
                        sum_y += clust_points[ii].y;
                    }
                    double c_x = sum_x / clust_points.size();
                    double c_y = sum_y / clust_points.size();
                    //double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
                    //double c_x = sum_x / x.size();
                    //double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
                    //double c_y = sum_y / y.size();
                    centroids.push_back({c_x, c_y});

                }
                // New object
                clust_points.clear();
                clust_points.push_back(Point({cart_x[i],cart_y[i]}));
                //x.clear();
                //y.clear();
                //x.push_back(cart_x[i]);
                //y.push_back(cart_y[i]);
            } else {
                // Add point to object
                clust_points.push_back(Point({cart_x[i],cart_y[i]}));
                //x.push_back(cart_x[i]);
                //y.push_back(cart_y[i]);
            }
        }
    }

    // Last cluster
    std::vector<Point> reduced_points;
    //for (int ii=0; ii<x.size(); ii++) {
    //    all_points.push_back(Point({x[ii],y[ii]}));
    //}
    RamerDouglasPeucker(clust_points, EPSILON, reduced_points);
    points.push_back(clust_points);
    filt_points.push_back(reduced_points);
    double sum_x = 0., sum_y = 0.;
    for (int ii=0; ii<clust_points.size(); ii++) {
        sum_x += clust_points[ii].x; 
        sum_y += clust_points[ii].y;
    }
    double c_x = sum_x / clust_points.size();
    double c_y = sum_y / clust_points.size();
    //double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
    //double c_x = sum_x / x.size();
    //double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
    //double c_y = sum_y / y.size();
    centroids.push_back({c_x, c_y});

    // Possible merge of the first and last clusters
    double dist = std::hypot(centroids.back().x - centroids.front().x, centroids.back().y - centroids.front().y);
    //double dist = sqrt( pow(centroids.back().x - centroids.front().x, 2) + pow(centroids.back().y - centroids.front().y, 2) );
    if (dist < CLUST_TOL) {
        points.back().insert(std::end(points.back()), std::begin(points[0]), std::end(points[0]));
        points[0] = points.back();
        points.erase(points.end() - 1);
        filt_points.back().insert(std::end(filt_points.back()), std::begin(filt_points[0]), std::end(filt_points[0]));
        filt_points[0] = filt_points.back();
        filt_points.erase(filt_points.end() - 1);
        centroids[0].x = (centroids.back().x + centroids.front().x)/2.0;
        centroids[0].y = (centroids.back().y + centroids.front().y)/2.0;
        centroids.erase(centroids.end() - 1);
    }
}

int hungarian_distance(Point const & p1, Point const & p2) {
    return std::hypot(p1.x - p2.x, p1.y - p2.y)*100;
}

void computeCostMatrix(std::vector<std::vector<int>> & matrix, std::vector<Point> const & _detections,  std::vector<cluster> const & _tracks) {
    int m = _detections.size();
    int n = _tracks.size();
    int size = m + n;
    matrix.resize(size, std::vector<int>(size));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            matrix[i][j] = hungarian_distance(_tracks[i].pos, _detections[j]); //residualsCov[i]);
        }
        for (int j=m; j<size; ++j) {
            matrix[i][j] = NO_ASSOCIATION;
        }
        matrix[i][m+i] = COST_UNASSOCIATED_TRACK;
    }

    for (int i=n; i<size; ++i) {
        for (int j=0; j<m; ++j) {
            matrix[i][j] = NO_ASSOCIATION;
        }
        matrix[i][i-n] = COST_UNASSOCIATED_DETECTION;

        for (int j=m; j<size; ++j) {
            matrix[i][j] = 0;
        }
    }
}

void detectionToTrackAssignment(std::vector<Point> const & _detections, std::vector<cluster> const & _tracks, std::vector<std::pair<int, int>> & _assignments, std::vector<int> & _unassignedTracks, std::vector<int> & _unassignedDetections) {
    int n = _tracks.size();
    int m = _detections.size();
    std::vector<bool> associatedTracks(n, false);
    std::vector<bool> associatedDetections(m, false);

    if (n>0 && m>0) {
        std::vector<std::vector<int> > costMatrix;
        computeCostMatrix(costMatrix, _detections, _tracks);

        // Convert to Matrix 
        Std2dVectorAdapter<int> adapter;
        Matrix<int> associationMatrix = adapter.convertToMatrix(costMatrix);

        // Apply Munkres algorithm to matrix.
        Munkres<int> munkres;
        munkres.solve(associationMatrix);

        for (int i=0; i<n; ++i) {
            for (int j=0; j<m; ++j) {
                if (associationMatrix(i,j) == 0) {
                    associatedTracks[i] = 1;
                    associatedDetections[j] = 1;
                    _assignments.push_back(std::make_pair(i,j)); // track -> detection
                }
            }
        }
    }

    for (int i=0; i<n; ++i) {
        if (!associatedTracks[i]) {
            _unassignedTracks.push_back(i);
        }
    }

    for (int i=0; i<m; ++i) {
        if (!associatedDetections[i]) {
            _unassignedDetections.push_back(i);
        }
    }
}

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


int main(int argc, char ** argv) {
    signal(SIGINT, ctrlc);

     if (argc<2) {
        std::cout << "Usage: " << argv[0] << " <paramsFile>" << std::endl;
        return 0;
    }

    std::cout << "[Initialization...]" << std::endl;
    ParamsProvider::init(argv[1]);

    std::string subAddressLoc    = ParamsProvider::getInstance().getSubFullAddress("loc");
    std::string subAddressOdom   = ParamsProvider::getInstance().getSubFullAddress("odom");
    std::string subAddress3D     = ParamsProvider::getInstance().getSubFullAddress("3Dcam");
    std::string subAddressLidar  = ParamsProvider::getInstance().getSubFullAddress("lidar");
    
    std::string pubAddressHumPos  = ParamsProvider::getInstance().getPubFullAddress("humPos");
    std::string pubAddressCluster = ParamsProvider::getInstance().getPubFullAddress("cluster");
    
    std::string topicLoc     = ParamsProvider::getInstance().getTopic("loc");
    std::string topicOdom    = ParamsProvider::getInstance().getTopic("odom");
    std::string topic3D      = ParamsProvider::getInstance().getTopic("3Dcam");
    std::string topicLidar   = ParamsProvider::getInstance().getTopic("lidar");
    
    std::string topicHumPos  = ParamsProvider::getInstance().getTopic("humPos");
    std::string topicCluster = ParamsProvider::getInstance().getTopic("cluster");

    ////////////////////////////////////////////////////////////////////////////
    // PUBLISHERS
    ////////////////////////////////////////////////////////////////////////////
    
    Publisher pubHumPos(pubAddressHumPos);
    Publisher pubCluster(pubAddressCluster);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO LIDAR
    ////////////////////////////////////////////////////////////////////////////
    
    struct RawClusters {
        std::vector<Point> centroids;
        std::vector<std::vector<Point>> points;
        std::vector<std::vector<Point>> filt_points;
        long long int timestamp = 0;
        std::mutex mtx;
    };

    Subscriber subLidar;
    RawClusters rawClusters;
    
    // Precompute LIDAR data angles
    std::vector<double> scan_angle(VECTOR_SIZE);
    for (int i=0; i<VECTOR_SIZE; ++i) {
        scan_angle[i] = i*0.5 + OFFSET_DEGS;;
    }

    subLidar.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        try {
            json jldr = json::parse(std::string(buf, size));
                    
            // Get lidar data
            std::vector<double> scan_dist = jldr.at("data");
            int size = scan_dist.size();
            ASSERT(VECTOR_SIZE == size);

            double lidar_x[VECTOR_SIZE];
            double lidar_y[VECTOR_SIZE];
            std::vector<Point> centroids;
            std::vector<std::vector<Point>> points;
            std::vector<std::vector<Point>> filt_points;
            
            // Polar to cartesian coordinates
            polar2cartesian(scan_dist.data(), scan_angle.data(), lidar_x, lidar_y, size);
            
            // Cluster and minimize lidar data
            clusterData(lidar_x, lidar_y, VECTOR_SIZE, centroids, points, filt_points);

            {
                std::unique_lock<std::mutex> lck(rawClusters.mtx);
                rawClusters.centroids = centroids;
                rawClusters.points = points;
                rawClusters.filt_points = filt_points;
                //rawClusters.angles = angles;
                rawClusters.timestamp = timeNow();
            }
        }
        catch(std::exception &e){
            std::cerr << "Error parsing lidar data: " << e.what() << std::endl;
        }
    });
    subLidar.start(subAddressLidar, topicLidar);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO CAMERA
    ////////////////////////////////////////////////////////////////////////////

    struct TimedPose {
        double x = 0, y = 0;
        long long int timestamp = 0;
        std::mutex mtx;
    };

    Subscriber sub3D;
    TimedPose track3D;
    sub3D.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        try {
            json jobj = json::parse(std::string(buf, size));
            long long int timestamp = jobj.at("ts");
            double x = jobj.at("x"); 
            double y = jobj.at("y");
        
            {
                std::unique_lock<std::mutex> lck(track3D.mtx);
                track3D.x = x; 
                track3D.y = y;
                track3D.timestamp = timestamp;

        
            }
        }
        catch(std::exception &e) {
            std::cerr << "Error parsing 3D camera data: " << e.what() << std::endl;
        }
    });
    sub3D.start(subAddress3D, topic3D);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO ODOM
    ////////////////////////////////////////////////////////////////////////////

    struct Odometry {
        double x, y, theta;
        double vx, vy, vtheta;
        long long int ts;
        std::mutex mtx;
    };

    Subscriber subOdom;
    Odometry odomData;
    subOdom.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        nlohmann::json j;

        try {
            j = nlohmann::json::parse(std::string(buf, size));   
            double x, y, theta;
            //double vx, vy, vtheta;
            double v, omega;
            long long int ts;
            ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            
            //x = j["pose"]["pose"]["position"]["x"]; 
            //y = j["pose"]["pose"]["position"]["y"];
            x = j["x"];
            y = j["y"];
            theta = j["theta"];

            //Quaternion q;
            //q.x = j["pose"]["pose"]["orientation"]["x"];
            //q.y = j["pose"]["pose"]["orientation"]["y"];
            //q.z = j["pose"]["pose"]["orientation"]["z"];
            //q.w = j["pose"]["pose"]["orientation"]["w"];
            //EulerAngles angles = ToEulerAngles(q);
            //theta = angles.yaw;
                    
            //vx = j["twist"]["twist"]["linear"]["x"];
            //vy = j["twist"]["twist"]["linear"]["y"];
            //vtheta = j["twist"]["twist"]["angular"]["z"];
            v = j["v"];
            omega = j["omega"];
            
            {
                std::unique_lock<std::mutex> lock(odomData.mtx);
                odomData.x = x;
                odomData.y = y;
                odomData.theta = theta;
                odomData.vx = v; //vx;
                odomData.vy = 0; //vy;
                odomData.vtheta = omega;//vtheta;
                odomData.ts = ts;
            }

        }
        catch(std::exception &e) {
            std::cerr << "Error parsing odom data: " << e.what() << std::endl;
        }
    });
    subOdom.start(subAddressOdom, topicOdom);

    ////////////////////////////////////////////////////////////////////////////
    // SUBSCRIBER TO LOC
    ////////////////////////////////////////////////////////////////////////////

    struct Pose {
        double x, y, theta;
        long long int ts;
        std::mutex mtx;
    };

    Subscriber subLoc;
    Pose locData;
    subLoc.register_callback([&](const char *topic, const char *buf, size_t size, void *data) {
        nlohmann::json j;
        
        try {
            j = nlohmann::json::parse(std::string(buf, size));
            double x, y, theta;
            long long int ts;
            x = j.at("loc_data").at("x");
            y = j.at("loc_data").at("y");
            theta = j.at("loc_data").at("theta");
            ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            {
                std::unique_lock<std::mutex> lock(locData.mtx);
                locData.x = x;
                locData.y = y;
                locData.theta = theta;
                locData.ts = ts;
            }
        }
        catch(std::exception &e)  {
            std::cerr << "\"" << std::string(buf, size) << "\"" << std::endl;
            std::cerr << "error parsing loc data: " << e.what() << std::endl;
        }
    });
    subLoc.start(subAddressLoc, topicLoc);

    ////////////////////////////////////////////////////////////////////////////
    // PROCESSING
    ////////////////////////////////////////////////////////////////////////////

    std::vector<cluster> Clusters;
    int NXT_ID = 1;

    // double pose_x = 0;
    // double pose_y = 0;
    // loc associated with pose
    // double loc_x = 0;
    // double loc_y = 0;
    // double loc_theta = 0;
    
    enum State { INIT, TRACK };
    State state = INIT; // 0: init, 1: track
    int cnt_leader_rejection = 0;
    
    // auto tryUpdateState = [&](double x_try, double y_try) {
    //     if (std::hypot(x_try-pose_x, y_try-pose_y)<TIME_TOL) {
    //         updateState(x_try, y_try);
    //     }
    //     else {
    //         fail();      
    //     }
    // };

    // auto findNearest = [](const RawClusters& clusters, double x_q, double y_q) {
    //     double d_min = 1e9;
    //     int idx_min  = -1;

    //     for (int i=0; i < clusters.clusters.size(); i++) {
    //         double dist = std::pow(x_q - clusters.clusters[i].x, 2) + std::pow(y_q - clusters.clusters[i].y, 2);
    //         if (dist < d_min) {
    //             d_min = dist;
    //             idx_min = i;
    //         }
    //     }

    //     return idx_min;
    // };

    std::cout << "[Looping...]" << std::endl;
    while (1) {
        
        // Exit
        if (ctrl_c_pressed) {
            std::cout << "CTRL-C PRESSED. TERMINATING." << std::endl;
            break;
        }

        Pose locDataC;
        {
            std::unique_lock<std::mutex> lock(locData.mtx);
            locDataC.x = locData.x;
            locDataC.y = locData.y;
            locDataC.theta = locData.theta;
            locDataC.ts = locData.ts;
        }

        Odometry odomDataC;
        {
            std::unique_lock<std::mutex> lock(odomData.mtx);
            odomDataC.x     = odomData.x;
            odomDataC.y     = odomData.y;
            odomDataC.theta = odomData.theta;
            odomDataC.vx     = odomData.vx;
            odomDataC.vtheta = odomData.vtheta;
            odomDataC.ts    = odomData.ts;
        }
        
        // roto-translate pose_x, pose_y to global according to (their) ROBOT loc
        // double cost = std::cos(loc_theta);
        // double sint = std::sin(loc_theta);
        // double pose2Dabs_x = cost*pose_x - sint*pose_y + loc_x;
        // double pose2Dabs_y = sint*pose_x + cost*pose_y + loc_y;
        // add ROBOT movement
        // double dx = pose2Dabs_x - locDataC.x;
        // double dy = pose2Dabs_y - locDataC.y;
        // roto-translate pose_x, pose_y to relative according to new (their) ROBOT loc
        // cost = std::cos(locDataC.theta);
        // sint = std::sin(locDataC.theta);
        // double pose2Drel_x = cost*dx + sint*dy;
        // double pose2Drel_y = -sint*dx + cost*dy;

        // pose_x = pose2Drel_x;
        // pose_y = pose2Drel_y;      
        
        // loc_x = locDataC.x;
        // loc_y = locDataC.y;
        // loc_theta = locDataC.theta;

        TimedPose track3DC;
        {
            std::unique_lock<std::mutex> lck(track3D.mtx);
            track3DC.x = track3D.x;
            track3DC.y = track3D.y;
            track3DC.timestamp = track3D.timestamp;
        }

        RawClusters rawClustersC;
        {
            std::unique_lock<std::mutex> lck(rawClusters.mtx);
            rawClustersC.centroids = rawClusters.centroids;
            rawClustersC.points = rawClusters.points;
            rawClustersC.filt_points = rawClusters.filt_points;
            rawClustersC.timestamp = rawClusters.timestamp;
        }

        long long int currTs = timeNow();
        
        // Roto-translate clusters to relative according to new ROBOT pos
        for (cluster & c: Clusters) {
            //double dx = c.pos.x - locDataC.x;
            //double dy = c.pos.y - locDataC.y;
            //double cost = std::cos(locDataC.theta);
            //double sint = std::sin(locDataC.theta);
      
            double dx = c.pos.x - odomDataC.x;
            double dy = c.pos.y - odomDataC.y;
            double cost = std::cos(odomDataC.theta);
            double sint = std::sin(odomDataC.theta);
      
            c.pos.x =  cost*dx + sint*dy;
            c.pos.y = -sint*dx + cost*dy;   
        }
        
        // Hungarian data association
        std::vector<std::pair<int, int>> assignments;
        std::vector<int> unassignedTracks;
        std::vector<int> unassignedDetections;
        detectionToTrackAssignment(rawClustersC.centroids, Clusters, assignments, unassignedTracks, unassignedDetections);
        
        // Update tracked clusters
        for (auto const & ass: assignments) {
            int idx_t = ass.first;
            int idx_d = ass.second;
            Clusters[idx_t].vel = (rawClustersC.centroids[idx_d] - Clusters[idx_t].pos)/(currTs-Clusters[idx_t].timestamp);
            Clusters[idx_t].pos = rawClustersC.centroids[idx_d];
            Clusters[idx_t].points = rawClustersC.points[idx_d];
            Clusters[idx_t].filt_points = rawClustersC.filt_points[idx_d];
            //Clusters[idx_t].angles = rawClustersC.angles[idx_d];
            Clusters[idx_t].timestamp = currTs;
            Clusters[idx_t].cnt_no_det = 0;
        }

        // Add new clusters
        for (auto const & idx_d: unassignedDetections) {
            cluster newCluster;
            newCluster.pos = rawClustersC.centroids[idx_d];
            newCluster.vel = Point({0, 0});
            newCluster.points = rawClustersC.points[idx_d];
            newCluster.filt_points = rawClustersC.filt_points[idx_d];
            //newCluster.angles = rawClustersC.angles[idx_d];
            newCluster.id = NXT_ID++;
            newCluster.ctype = cluster::UNKNOWN;
            newCluster.timestamp = currTs;
            Clusters.emplace_back(newCluster);
        }
        
        bool validCamera = std::abs(currTs-track3DC.timestamp)<VALID_TIME;
        // bool validClusters = !rawClustersC.clusters.empty() && std::abs(currTs-rawClustersC.timestamp)<VALID_TIME;

        if (validCamera && state==INIT) {
            int cid = -1;
            double dMin = std::numeric_limits<double>::infinity();
            for (int i=0; i<Clusters.size(); ++i) {
                cluster & c = Clusters[i];
                double dst = std::hypot(c.pos.x-track3DC.x, c.pos.y-track3DC.y);
                if (dst<dMin) {
                    dMin = dst;
                    cid = i;
                }
            }
            if (cid>=0 && dMin<DIST_PROMOTION) {
                Clusters[cid].cnt_prom++;
                if (Clusters[cid].cnt_prom>=LEADER_PROMOTION) {
                    Clusters[cid].ctype = cluster::LEADER;
                    for (int j=0; j<Clusters.size(); ++j) {
                        if (j!=cid) {
                            Clusters[j].ctype = cluster::OBSTACLE;
                        }
                        Clusters[j].cnt_prom = 0;
                    }
                    state = TRACK;
                }
                else {
                    for (int j=0; j<Clusters.size(); ++j) {
                        if (j!=cid) {
                            Clusters[j].cnt_prom = 0;
                        }
                    }
                }
            }
            else {
                for (int j=0; j<Clusters.size(); ++j) {
                    Clusters[j].cnt_prom = 0;
                }
            }
        }
        else if (validCamera && state==TRACK) {
            int lid = -1;
            for (int i=0; i<Clusters.size(); ++i) {
                if (Clusters[i].ctype == cluster::LEADER) {
                    lid = i;
                    break;
                }
            }

            ASSERT(lid>=0);
            double dst = std::hypot(Clusters[lid].pos.x-track3DC.x, Clusters[lid].pos.y-track3DC.y);
            if (dst<D_THRESH) {
                for (int i=0; i<Clusters.size(); ++i) {
                    if (Clusters[i].ctype == cluster::UNKNOWN) {
                        Clusters[i].ctype = cluster::OBSTACLE;
                    }
                }
                cnt_leader_rejection = 0;
            }
            else {
                cnt_leader_rejection++;
                if (cnt_leader_rejection >= LEADER_REJECTION) {
                    state = INIT;
                    for (int i=0; i<Clusters.size(); ++i) {
                        Clusters[i].ctype = cluster::UNKNOWN;
                    }
                    cnt_leader_rejection = 0;
                }
            }

        }
        
        // Delete clusters
        bool leaderDiscarded = false;
        for (int i=unassignedTracks.size()-1; i>=0; --i) {
            int idx_t = unassignedTracks[i];
            ++Clusters[idx_t].cnt_no_det;
            if (Clusters[idx_t].cnt_no_det >= FAIL_DETECTION) {
                if (Clusters[idx_t].ctype == cluster::LEADER) 
                    leaderDiscarded = true;
                Clusters.erase(Clusters.begin()+idx_t);
            }
        }
        
        if (leaderDiscarded) {
            state = INIT;
            for (int i=0; i<Clusters.size(); ++i) {
                Clusters[i].ctype = cluster::UNKNOWN;
            }
            cnt_leader_rejection = 0;
        }
    
        // bool validCamera = std::abs(currTs-track3DC.timestamp)<VALID_TIME;
        // bool validClusters = !rawClustersC.clusters.empty() && std::abs(currTs-rawClustersC.timestamp)<VALID_TIME;

        // if (!validClusters && !validCamera) {
        //     if (state==INIT) {} // do nothing
        //     else if (state==TRACK) fail();
        // }
        // else if (!validClusters && validCamera) {
        //     if (state==INIT) {} // do nothing
        //     else if (state==TRACK) {
        //         tryUpdateState(track3DC.x, track3DC.y);
        //     }
        // }
        // else if (validClusters && !validCamera) {
        //     if (state==INIT) {} // do nothing
        //     else if (state==TRACK) {
        //         int idx = findNearest(rawClustersC, pose_x, pose_y);
        //         tryUpdateState(rawClustersC.clusters[idx].x, rawClustersC.clusters[idx].y);
        //     }
        // }
        // else if (validClusters && validCamera) {
        //     if (state==INIT) {
        //         int idx = findNearest(rawClustersC, track3DC.x, track3DC.y);
        //         if (std::hypot(rawClustersC.clusters[idx].x-track3DC.x, rawClustersC.clusters[idx].y-track3DC.y)<DIST_TOL) {
        //             state = TRACK;
        //             updateState(0.5*(rawClustersC.clusters[idx].x+track3DC.x), 0.5*(rawClustersC.clusters[idx].y+track3DC.y));
        //         }
        //         else {} // do nothing
        //     }
        //     else if (state==TRACK) {

        //         int idx_prev = findNearest(rawClustersC, pose_x, pose_y);    // nearest cluster w.r.t. previous known pose
        //         int idx = findNearest(rawClustersC, track3DC.x, track3DC.y); // nearest cluster w.r.t. camera
                      
        //         double x_cam, y_cam;
        //         if (std::hypot(rawClustersC.clusters[idx].x-track3DC.x, rawClustersC.clusters[idx].y-track3DC.y)<DIST_TOL) {
        //             x_cam = 0.5*(rawClustersC.clusters[idx].x+track3DC.x);
        //             y_cam = 0.5*(rawClustersC.clusters[idx].y+track3DC.y);
        //         }
        //         else {
        //             x_cam = track3DC.x;
        //             y_cam = track3DC.y;
        //         }

        //         double d_ldr_prev = std::hypot(rawClustersC.clusters[idx_prev].x-pose_x, rawClustersC.clusters[idx_prev].y-pose_y);
        //         double d_cam_prev = std::hypot(x_cam-pose_x, y_cam-pose_y);
        //         double xStar = 0; // (d_ldr_prev < d_cam_prev) ? rawClustersC.clusters[idx_prev].x : x_cam;
        //         double yStar = 0; //(d_ldr_prev < d_cam_prev) ? rawClustersC.clusters[idx_prev].y : y_cam;

        //         //double xStar, yStar;
        //         //if (std::hypot(trackLdrC.xs[cvidx]-track3DC.x, trackLdrC.ys[idx]-track3DC.y)<DIST_TOL) {
        //         //    xStar = 0.5*(trackLdrC.xs[idx]+track3DC.x);
        //         //    yStar = 0.5*(trackLdrC.ys[idx]+track3DC.y);
        //         //}
        //         //else {
        //         //    int idx = findNearest(trackLdrC, pose_x, pose_y); // best lidar w.r.t. previous known pose
        //         //    double d_ldr_track = std::hypot(trackLdrC.xs[idx]-pose_x, trackLdrC.ys[idx]-pose_y);
        //         //    double d_cam_track = std::hypot(track3DC.x-pose_x, track3DC.y-pose_y);
        //         //    xStar = (d_ldr_track < d_cam_track) ? trackLdrC.xs[idx] : track3DC.x;
        //         //    yStar = (d_ldr_track < d_cam_track) ? trackLdrC.ys[idx] : track3DC.y;
        //         //}
        //         tryUpdateState(xStar, yStar);
        //     } 
        // }

        // Publish jsons
        std::string jmsg;

        double pose_x = 0, pose_y = 0;
        if (state == TRACK) {
            for (int i=0; i<Clusters.size(); ++i) {
                if (Clusters[i].ctype == cluster::LEADER) {
                    pose_x = Clusters[i].pos.x;
                    pose_y = Clusters[i].pos.y;
                    break;
                }
            }
        }

        json jHumPos;
        jHumPos["ts"] = currTs;
        jHumPos["valid"] = state==TRACK;
        jHumPos["x"] = pose_x;
        jHumPos["y"] = pose_y;        
        jmsg = jHumPos.dump();
        pubHumPos.send(topicHumPos.c_str(), jmsg.c_str(), jmsg.size());

        /*
        json jtmp = json::array();
        for (auto const & cluster: rawClustersC.clusters) {
            json jc;
            jc["x"] = cluster.x;
            jc["y"] = cluster.y;
            jtmp.push_back(jc);
        }

        json jClusters;
        jClusters["ts"] = timeNow();
        jClusters["clusters"] = jtmp;
        jmsg = jClusters.dump();
        pubCluster.send(pubTopicCluster.c_str(), jmsg.c_str(), jmsg.size());
        */

        json jtmp = json::array();
        for (auto const & cluster: Clusters) {
            json jc;
            jc["x"] = cluster.pos.x;
            jc["y"] = cluster.pos.y;
            jc["vx"] = cluster.vel.x;
            jc["vy"] = cluster.vel.y;
            for (int i=0; i<cluster.points.size(); i++) {
                jc["points_x"][i] = cluster.points[i].x;
                jc["points_y"][i] = cluster.points[i].y;
            }
            for (int i=0; i<cluster.filt_points.size(); i++) {
                jc["filt_points_x"][i] = cluster.filt_points[i].x;
                jc["filt_points_y"][i] = cluster.filt_points[i].y;
            }
            //jc["min_angle"] = cluster.angles.x;
            //jc["max_angle"] = cluster.angles.y;
            jc["id"] = cluster.id;
            jc["type"] = cluster.ctype;
            jc["visible"] = cluster.cnt_no_det == 0;
            jtmp.push_back(jc);
        }

        json jClusters;
        jClusters["ts"] = timeNow();
        jClusters["clusters"] = jtmp;
        jmsg = jClusters.dump();
        pubCluster.send(topicCluster.c_str(), jmsg.c_str(), jmsg.size());
        
        // Roto-translate clusters to absolute according to new ROBOT pos
        for (cluster & c: Clusters) {
            //double cost = std::cos(locDataC.theta);
            //double sint = std::sin(locDataC.theta);
            double cost = std::cos(odomDataC.theta);
            double sint = std::sin(odomDataC.theta);
            double pose2Drel_x = c.pos.x;
            double pose2Drel_y = c.pos.y;

            //c.pos.x = cost*pose2Drel_x - sint*pose2Drel_y + locDataC.x;
            //c.pos.y = sint*pose2Drel_x + cost*pose2Drel_y + locDataC.y;
            c.pos.x = cost*pose2Drel_x - sint*pose2Drel_y + odomDataC.x;
            c.pos.y = sint*pose2Drel_x + cost*pose2Drel_y + odomDataC.y;            
        }
        
        delay(100);

    }
    
    return 0;
}

EulerAngles ToEulerAngles(Quaternion const & q) {
  EulerAngles angles;

  double test = q.x*q.y + q.z*q.w;
  if (test > 0.499) { // singularity at north pole
    angles.roll = 2 * std::atan2(q.x,q.w);
    angles.pitch = M_PI/2;
    angles.yaw = 0;
    return angles;
  }
  if (test < -0.499) { // singularity at south pole
    angles.roll = -2 * std::atan2(q.x,q.w);
    angles.pitch = - M_PI/2;
    angles.yaw = 0;
    return angles;
  }
  double sqx = q.x*q.x;
  double sqy = q.y*q.y;
  double sqz = q.z*q.z;
  angles.roll = std::atan2(2*q.y*q.w-2*q.x*q.z , 1 - 2*sqy - 2*sqz);
  angles.pitch = std::asin(2*test);
  angles.yaw = std::atan2(2*q.x*q.w-2*q.y*q.z , 1 - 2*sqx - 2*sqz);

  return angles;
}