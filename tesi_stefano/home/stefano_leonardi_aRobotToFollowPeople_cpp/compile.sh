#!/bin/bash

echo "NB: the building folder is not removed because it can contains submodules."

#the check is done for the docker container
if ! [[ -d build ]] 
then
	echo "build            folder do NOT exists yet."
	mkdir build
fi
cd build

if ! [[ -d buildLinux ]] 
then
	echo "build/buildLinux folder do NOT exists yet."
	mkdir buildLinux
fi
cd buildLinux

cmake ../../
make

cd .. && cd ..

