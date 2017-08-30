
#ifndef SIMULATION_IO_HPP
#define SIMULATION_IO_HPP


#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>

#include "LinearMath/btVector3.h"
#include "H5Cpp.h"
#include "hdf5.h"

struct sim_data{
	std::vector<std::vector<std::vector<float>>> M;
	std::vector<std::vector<std::vector<float>>> F;
	std::vector<std::vector<std::vector<float>>> X;
	std::vector<std::vector<std::vector<float>>> Y;
	std::vector<std::vector<std::vector<float>>> Z;
	std::vector<float> T;

};


void save_data(sim_data* data, std::string filename = "../output/test.h5");
void addToDataset3D(std::vector<std::vector<std::vector<float>>> M, H5std_string dataname, H5::H5File file);
void addToDataset1D(std::vector<float> data, H5std_string dataname, H5::H5File file);

#endif //SIMULATION_IO_HPP