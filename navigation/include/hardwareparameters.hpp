#ifndef HARDWAREPARAMETERS_H
#define HARDWAREPARAMETERS_H

#include <string>

struct HardwareParameters {
public:
  std::string hardwareServer        = "tcp://10.196.80.132:5601";

  std::string localizationPublisher = "tcp://10.196.80.135:9207";
  std::string odomPublisher         = "tcp://10.196.80.135:9307";
  std::string lidarPublisher        = "tcp://10.196.80.135:7500";

  std::string localizationTopic     = "POS";
  std::string odomTopic             = "ODOM";
  std::string lidarTopic            = "LIDAR";

  double safetyWidth = 0.4;
  double criticalDistance = 0.4;

  double ha = 0.25; //0.1; //distanza x della punta del veicolo rispetto al suo sistema di riferimento
  double hb = 0.25; //0.7; //distanza x della coda del veicolo rispetto al suo sistema di riferimento
  double width = 0.5; //larghezza veicolo
  //double length = 0.5; unused

};

#endif // HARDWAREPARAMETERS_H
