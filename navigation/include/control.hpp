#pragma once

#include <cmath>

#include <ClothoidList.hh>
#include "hardwareglobalinterface.hpp"

double computeControl(double DT, 
                      G2lib::ClothoidList const & path,
                      double & vCmd,
                      double & omegaCmd);