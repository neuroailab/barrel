
#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#define PI 3.1415927

#define LINK_THETA_MAX PI/4
#define LINK_PHI_MAX PI/4
#define LINK_ZETA_MAX PI/360

#include <string>
#include <vector>

struct Parameters{
    
    int SOLVER;
	int DEBUG;
    int TEST;
    int OPT;
	int ACTIVE;
	int COLLIDE;
	float TEST_FORCE;

	float TIME_STEP;
	int NUM_STEP_INT;
	float TIME_STOP;
	int PRINT;
    
    std::vector<std::string> WHISKER_NAMES;
    float BLOW;
	int NO_CURVATURE;
	int NUM_UNITS;
	int NUM_LINKS;
	float DENSITY;
	float E_BASE;
	float E_TIP;
	float ZETA_BASE;
	float ZETA_TIP;

	float BT_RATIO;

	float WHISK_AMP;
	float WHISK_FREQ;

	std::vector<float> stiffness_x;
	std::vector<float> stiffness_y;
	std::vector<float> stiffness_z;
	std::vector<float> damping_x;
	std::vector<float> damping_y;
	std::vector<float> damping_z;

};

void set_default(Parameters* param);
std::vector<float> get_vector(int N, float value);
std::vector<float> stringToFloatVect(std::vector<std::string> vect_string);

#endif