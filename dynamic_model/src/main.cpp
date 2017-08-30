
/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2015 Google Inc. http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#include "Simulation.hpp"


#include "../CommonInterfaces/CommonExampleInterface.h"
#include "../CommonInterfaces/CommonGUIHelperInterface.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionShapes/btCollisionShape.h"
#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"


#include "LinearMath/btTransform.h"
#include "LinearMath/btHashMap.h"

#include <iostream>
#include <boost/program_options.hpp>



void objFunction(Parameters* param);


int main(int argc, char* argv[])
{

	Parameters* param = new Parameters();
	set_default(param);

  	try 
  	{ /** Define and parse the program options  */ 
	    namespace po = boost::program_options; 
	    po::options_description desc("Options"); 
	    desc.add_options() 
	      ("help,h", "Help screen")
	      ("SOLVER", po::value<int>(&param->SOLVER), "NNCG")
	      ("DEBUG", po::value<int>(&param->DEBUG), "debug on/off")
	      ("TEST", po::value<int>(&param->TEST), "test on/off")
	      ("OPT", po::value<int>(&param->OPT), "optimization on/off")
	      ("ACTIVE,a", po::value<int>(&param->ACTIVE), "active on/off")
	      ("COLLIDE", po::value<int>(&param->COLLIDE), "collide on/off")
	      ("TEST_FORCE", po::value<float>(&param->TEST_FORCE), "test force")
	      ("TIME_STEP", po::value<float>(&param->TIME_STEP), "time step")
	      ("NUM_STEP_INT", po::value<int>(&param->NUM_STEP_INT), "number of internal steps")
	      ("TIME_STOP", po::value<float>(&param->TIME_STOP), "time stop")
	      ("PRINT", po::value<int>(&param->PRINT), "print simulation output")
	      
	      ("WHISKER_NAMES", po::value<std::vector<std::string> >(&param->WHISKER_NAMES)->multitoken(), "whisker names to simulate")
	      ("BLOW,b", po::value<float>(&param->BLOW), "whisker curvature on/off")
	      ("NO_CURVATURE", po::value<int>(&param->NO_CURVATURE), "whisker curvature on/off")
	      ("NUM_UNITS", po::value<int>(), "number of units")
	      ("BT_RATIO", po::value<float>(&param->BT_RATIO), "ratio base/tip")
	      ("STIFFNESS", po::value<float>(), "global stiffness")
	      ("DAMPING", po::value<float>(), "global damping")
	      ("WHISK_AMP", po::value<float>(), "whisk amplitude")
	      ("WHISK_FREQ", po::value<float>(&param->WHISK_FREQ), "whisk frequency")
	      
	      ("stiffness_x", po::value< std::vector<float> >(&param->stiffness_x)->multitoken(), "stiffness for x axis")
	      ("stiffness_y", po::value< std::vector<float> >(&param->stiffness_y)->multitoken(), "stiffness for y axis")
	      ("stiffness_z", po::value< std::vector<float> >(&param->stiffness_z)->multitoken(), "stiffness for z axis")
	      ("damping_x", po::value< std::vector<float> >(&param->damping_x)->multitoken(), "damping for x axis")
	      ("damping_y", po::value< std::vector<float> >(&param->damping_y)->multitoken(), "damping for y axis")
	      ("damping_z", po::value< std::vector<float> >(&param->damping_z)->multitoken(), "damping for z axis");
	 	

	 	
	    po::variables_map vm; 


	    try { 
		    po::store(po::parse_command_line(argc, argv, desc), vm); // can throw 
		 	po::notify(vm);

		 	if ( vm.count("help")  ) { 
		        std::cout << "Bullet Whisker Simulation" << std::endl 
		                  << desc << std::endl; 
		        return 0; 
		    } 

		    if (vm.count("WHISK_AMP")){
		    	param->WHISK_AMP = vm["WHISK_AMP"].as<float>()*PI/180.;
		      	// std::cout << "Solver: " << vm["solver"].as<int>() << '\n';
		    }
		    else if (vm.count("NUM_UNITS")){
		    	param->NUM_UNITS = vm["NUM_UNITS"].as<int>();
		    	param->NUM_LINKS = param->NUM_UNITS - 1;
		      	// std::cout << "Solver: " << vm["solver"].as<int>() << '\n';
		    }
		    else if (vm.count("STIFFNESS")){
		    	float k = vm["STIFFNESS"].as<float>();
		    	param->stiffness_x = get_vector(param->NUM_UNITS,k);
				param->stiffness_y = get_vector(param->NUM_UNITS,k);
				param->stiffness_z = get_vector(param->NUM_UNITS,k);
		    }
		     else if (vm.count("DAMPING")){
		    	float d = vm["DAMPING"].as<float>();
		    	param->damping_x = get_vector(param->NUM_UNITS,d);
				param->damping_y = get_vector(param->NUM_UNITS,d);
				param->damping_z = get_vector(param->NUM_UNITS,d);
		    }
		    
	    	if (param->WHISKER_NAMES[0] == "ALL"){
	    		param->WHISKER_NAMES = {
	    			"LA0","LA1","LA2","LA3","LA4",
	    			"LB0","LB1","LB2","LB3","LB4","LB5",
	    			"LC0","LC1","LC2","LC3","LC4","LC5","LC6",
	    			"LD0","LD1","LD2","LD3","LD4","LD5","LD6",
	    			"LE1","LE2","LE3","LE4","LE5","LE6",
	    			"RA0","RA1","RA2","RA3","RA4",
	    			"RB0","RB1","RB2","RB3","RB4","RB5",
	    			"RC0","RC1","RC2","RC3","RC4","RC5","RC6",
	    			"RD0","RD1","RD2","RD3","RD4","RD5","RD6",
	    			"RE1","RE2","RE3","RE4","RE5","RE6"};
	    	}

	    	// for(int i=0;i<param->stiffness_x.size();i++){
	    	// 	std::cout << param->stiffness_x[i] << std::endl;
	    	// }

		  	DummyGUIHelper noGfx;

			CommonExampleOptions options(&noGfx);
			Simulation* simulation = SimulationCreateFunc(options);

			simulation->parameters = param; // save parameters in simulation object

			simulation->initPhysics();
			std::cout.precision(17);
			
			// run simulation
			do{
				simulation->stepSimulation();
			}while(!(simulation->exitFlag));

			std::cout << "Saving data..." << std::endl;

			sim_data* results = simulation->get_results();
			

			std::cout << "Exit simulation..." << std::endl;
			simulation->exitPhysics();

			delete simulation;
			std::cout << "Done." << std::endl;

			save_data(results);
			
	    } 
	    catch(po::error& e) 
	    { 
	      std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
	      std::cerr << desc << std::endl; 
	      return 1; 
	    } 
 // application code here // 
 
  	} 
  	catch(std::exception& e) 
  	{ 	std::cerr << "Unhandled Exception reached the top of main: "           << e.what() << ", application will now exit" << std::endl; 	return 2; 
 
  	} 

	
	return 0;
}
