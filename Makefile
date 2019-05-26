all: exe 

exe: process.cpp
	g++ -std=c++11 -o exe process.cpp `pkg-config opencv --cflags` `pkg-config opencv --libs`



