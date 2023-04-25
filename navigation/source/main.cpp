#include "robodogController.hpp"
#include "hardwareparameters.hpp"
#include "hardwareglobalinterface.hpp"

// HEADERS FOR DEBUGGING
#include "timer.hpp"
#include "Publisher.hpp"
#include "ClothoidList.hh"
#include "json.hpp"
using namespace nlohmann;


#include <atomic>
#include <signal.h>
#include <thread>
#include <chrono>

bool ctrlc_pressed = false;
void ctrlc_callback(int) {
    ctrlc_pressed = true;
}

// PATH SERIALIZATION FOR DEBUGGING
namespace G2lib {

    void to_json(json& j, const ClothoidList& cl) {
        j = json::array();
        for (int i=0; i<cl.numSegment(); ++i) {
            json jc(cl.get(i));
            j.push_back(jc);
        }
    }

    void to_json(json& j, const ClothoidCurve& cc) {
        j["x0"] = cc.xBegin();
        j["y0"] = cc.yBegin();
        j["t0"] = cc.thetaBegin();
        j["k0"] = cc.kappaBegin();
        j["dk"] = cc.dkappa();
        j["L"]  = cc.length();
    }

};


int main() {
    std::cout << "[Initialization...]" << std::endl;

    signal(SIGINT, ctrlc_callback);

    HardwareParameters hp;
    HardwareGlobalInterface::initialize(hp);
    RobodogController rc;
    rc.start();

    Timer pubTask;
    Publisher pubPath("tcp://*:3221");
    pubTask.registerCallback([&]() {
        // Get path
        std::pair<G2lib::ClothoidList, std::vector<int>> path = rc.getPath();
        // Construct and publish json
        json jobj;
        jobj["ts"] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        jobj["path"] = path.first;
        std::string jmsg = jobj.dump();
        pubPath.send("PATH", jmsg.c_str(), jmsg.size());
    });
    pubTask.start(100);

    std::cout << "[Looping...]" << std::endl;
    while (!ctrlc_pressed) {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);
    }

    std::cout << "[Terminating...]" << std::endl;
    rc.stop();
    pubTask.stop();

    return 0;
}