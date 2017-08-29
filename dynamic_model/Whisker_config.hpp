
#ifndef WHISKER_CONFIG_HPP
#define WHISKER_CONFIG_HPP

#include "Simulation_utility.hpp"

#include "LinearMath/btVector3.h"
#include "H5Cpp.h"
#include "hdf5.h"

#include <vector>
#include <string>


struct whisker_config{
	std::string id;
	int side;
	int row;
	int col;
	float L;
	float a;
	btVector3 base_pos;
	btVector3 base_rot;
};


whisker_config get_parameters(std::string wname);
float get_dzeta(int index);
float get_dphi(int index);

#endif // 