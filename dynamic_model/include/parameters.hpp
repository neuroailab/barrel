
#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include "LinearMath/btVector3.h"
#include <vector>

struct whisker_config {
	std::string id;
	float L;
	float a;
	btVector3 base_pos;
	btVector3 base_rot;
};


std::vector<whisker_config> whisker_parameters =
{ // for the vector
    { // for a whisker_config
    	"RA0",							// id
    	0.046402,							// L
        11.748268, 						// a
        btVector3(0.005359, -0.010331, -0.000461), 	// base_pos
        btVector3(5.549541, -0.459404, 1.613332)  	// base_rot (ratmap: [theta phi zeta])
    },
    {
        "RA1",							// id
    	0.038471,							// L
        17.032088, 						// a
        btVector3(0.005287, -0.007605, -0.000504), 	// base_pos
        btVector3(5.735375, -0.534912, 1.94093)  	// base_rot (ratmap: [theta phi zeta])
    }
};




#endif //