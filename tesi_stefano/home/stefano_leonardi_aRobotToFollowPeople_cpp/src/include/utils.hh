/** 
 *  @file   utils.hh
 *  @brief  The file containing all the utilities.
 *  @author Stefano Leonardi
 ***********************************************/
#pragma once
#include <iostream>
#include <vector>
#include <utility>

#include <opencv2/opencv.hpp>
#include <time.h>
#include <thread>
#include <chrono>
#include <ctime>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

using std::cout; 
using std::endl;
using std::pair;
using std::string;
using std::vector;
using std::ostream;

/** \brief Print to the console the name (code side) and the content of the given variable.
 *	\param[in] x The variable to be shown.
 */
#define watch(x) cout << (#x) << " is: " << (x) << endl



struct TimedPose {
	bool valid = false;
	float x = 0, y = 0;
	long long int timestamp = 0;
};



/** \brief Overriding of the operator << to show a vector to the console in the form of [el1, el2, el3].
 *	\tparam T The type of elements of the vector.
 *	\param[in] os The output stream where the custom print is written.
 *	\param[in] v The vector to be printed.
 *	\returns The modified output stream.
 */
template<typename T>
ostream & operator<<(ostream & os, const vector<T> & v){
	os << "[";
	if(v.size()>0){
		for(int i=0; i<v.size()-1; i++){
			os << v[i] << ", ";
		}
		os << v[v.size()-1];
	}
	return os << "]";
}

/** \brief Overriding of the operator << to show a pair to the console in the form of {el1, el2}.
 *	\tparam T1 The type of element 1 of the pair.
 *	\tparam T2 The type of element 2 of the pair.
 *	\param[in] os The output stream where the custom print is written.
 *	\param[in] p The pair to be printed.
 *	\returns The modified output stream.
 */
template<typename T1, typename T2>
ostream & operator<<(ostream & os, const pair<T1, T2> & p){
	os << "{" << p.first << ", " << p.second << "}";
	return os;
}

ostream & operator<<(ostream & os, const TimedPose & p);

/** \brief Show the type of the given variable.
 *	\tparam T The type of the variable.
 *	\param[in] v The input variable.
 */
template<typename T>
void variableType(const T & v){
	cout << typeid(v).name() << endl;
}

/** \brief Compute the actual time, measured in milliseconds since a certain date.
 *	\return The measured milliseconds.
 */
long long int timeNow();

/** \brief Sleep for the time specified.
 *	\param[in] millisec How many milliseconds to sleep.
 */
void sleepMil(int millisec);

/** \brief Generate a random number integer, if the function is called more than once the generated numebrs will be uniformally distributed.
 *	\param[in] vMin The lower bound of interval for the generated number.
 *	\param[in] vMax The upper bound of interval for the generated number.
 *	\return The generated number.
 */
int randUniform(int vMin, int vMax);

#include <sstream>
/** \brief   Convert a double to a string.
 *  \param[in] d The double to be converted.
 *  \param[in] precision The precision of the conversion. How many decimal digits should be preserved.
 *  \return The generated string.
 */
string doubleToString(double d, int precision = 2);

string trim(const string& str);

