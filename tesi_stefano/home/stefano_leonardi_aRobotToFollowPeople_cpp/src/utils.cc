#include "utils.hh"

ostream & operator<<(ostream & os, const TimedPose & p){
    os << "{" << p.timestamp << ": (" << p.x << ", " << p.y << ")}";
	return os;
}


long long int timeNow(){
    return (long long int)duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void sleepMil(int millisec){
    std::this_thread::sleep_for(std::chrono::milliseconds(millisec));
}

int randUniform(int vMin, int vMax){
	static cv::RNG rng(time(NULL));
	return (int)rng.uniform(vMin, vMax);
}


string doubleToString(double d, int precision){
    // Convert a double to a string with the given precision.
    double p = pow(10, precision);
    d = round(d * p) / p;
    std::ostringstream os;
    os << d;
    return os.str ();
}

string trim(const string& str)
{
    size_t first = str.find_first_not_of(" \n\r\t");
    if (string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(" \n\r\t");
    return str.substr(first, (last - first + 1));
}