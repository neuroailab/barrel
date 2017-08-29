
#include "Parameters.hpp"

// set default parameter values
void set_default(Parameters* param){

 	param->SOLVER = 1;		// select solver
	param->DEBUG = 0;		// enable debug mode
	param->OPT = 0;			// enable optimization mode
	param->TEST = 0;		// enable test mode
	param->ACTIVE = 0;		// enable active whisking mode
	param->COLLIDE = 0;		// enable collision object
	param->TEST_FORCE = 0.;	// set test force

	param->TIME_STEP = 1./250.;	// set time step
	param->NUM_STEP_INT = 25;	// set internal time step
	param->TIME_STOP = 1.;		// set simulation time
    param->PRINT = 0;			// set print out of results (necessary for optimization)

    param->WHISKER_NAMES = {"RA0"}; // select whiskers to simulate
    param->BLOW = 1;				// increase whisker diameter for better visualization (will affect dynamics!!)
	param->NO_CURVATURE = 0;		// disable curvature
	param->NUM_UNITS = 8;			// set number of units
	param->NUM_LINKS = 7;			// set number of links
	param->DENSITY = 1300;			// set whisker density
	param->E_BASE = 4.5;			// set young's modulus at whisker base
	param->E_TIP = 3.;				// set young's modulus at whisker tip
	param->ZETA_BASE = 1.;			// set damping coefficient zeta at whisker base
	param->ZETA_TIP = 0.8;			// set damping coefficient zeta at whisker tip

	param->BT_RATIO = 0.02;			// set base/tip radius ratio
	param->WHISK_AMP = 40.*PI/180;	// set whisking amplitude
	param->WHISK_FREQ = 1.;			// set whisking frequency (Hz)


	// set stiffness and damping for all axis
	param->stiffness_x = get_vector(param->NUM_UNITS,20000.);
	param->stiffness_y = get_vector(param->NUM_UNITS,20000.);
	param->stiffness_z = get_vector(param->NUM_UNITS,20000.);
	param->damping_x = get_vector(param->NUM_UNITS,200.);
	param->damping_y = get_vector(param->NUM_UNITS,200.);
	param->damping_z = get_vector(param->NUM_UNITS,200.);



}

// create vector
std::vector<float> get_vector(int N, float value){
	std::vector<float> vect;
	for(int i=0; i<N; i++){
		vect.push_back(value);
	}
	return vect;
}

// convert string to float vector - not used I think
std::vector<float> stringToFloatVect(std::vector<std::string> vect_string){
	std::string::size_type sz;
	std::vector<float> vect_float;
	for(int i=0; i<vect_string.size(); i++){
		vect_float.push_back(std::stof(vect_string[i],&sz));
	}
	return vect_float;
}

