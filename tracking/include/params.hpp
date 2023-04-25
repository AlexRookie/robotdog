#pragma once

#include "json.hpp"
#include <memory>
#include <fstream>

class ParamsProvider {
public:
	static void init(std::string filename) {
		if (ParamsProvider::instance) throw std::runtime_error("ParamsProvider already initialized.");
		ParamsProvider::instance = std::unique_ptr<ParamsProvider>(new ParamsProvider(filename));
	}

	static const ParamsProvider & getInstance() {
		if (!ParamsProvider::instance) throw std::runtime_error("ParamsProvider not initialized yet.");
		return *ParamsProvider::instance;
	}

	std::string getPubFullAddress(std::string name) const {
		std::string addr = "tcp://*";
		std::string portField = "port_" + name;
		std::string port = jobj.at(portField);
		return addr + ":" + port;
	}

	std::string getSubFullAddress(std::string name) const {
		std::string addrField = "address_" + name;
		std::string addr = jobj.at(addrField);
		std::string portField = "port_" + name;
		std::string port = jobj.at(portField);
		return addr + ":" + port;
	}

	std::string getTopic(std::string name) const {
		std::string topicField = "topic_" + name;
		return jobj.at(topicField);
	}

private:    
	ParamsProvider(std::string filename) {
		std::ifstream input(filename);
		if (!input.is_open()) {
			throw std::runtime_error("Failed to open file \"" + filename + "\"");
		}
		jobj = nlohmann::json::parse(input);
	}
	
	static std::unique_ptr<ParamsProvider> instance;
	
	nlohmann::json jobj;
};