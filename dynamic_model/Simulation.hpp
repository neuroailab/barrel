
#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "WhiskerArray.hpp"
#include "Simulation_utility.hpp"
#include "Simulation_IO.hpp"
// #include "LoadMeshFromSTL.hpp"

#include <iostream>

#include "btBulletDynamicsCommon.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "CommonInterfaces/CommonRigidBodyBase.h"
#include "CommonInterfaces/CommonParameterInterface.h"

class Simulation* SimulationCreateFunc(struct CommonExampleOptions& options);



class Simulation : public CommonRigidBodyBase
{

private: 
	float m_time;
	btVector3 gravity = btVector3(0,0,-9.8*SCALE);
	// btAlignedObjectArray<btJointFeedback*> m_jointFeedback;
	// btAlignedObjectArray<Whisker*> m_whiskerArray;
	
	WhiskerArray* whiskerArray;
	sim_data* data_dump = new sim_data();

	

public:

	Simulation(struct GUIHelperInterface* helper):CommonRigidBodyBase(helper){}
	virtual ~Simulation(){}
	virtual void initPhysics();
	virtual void exitPhysics();
	virtual void stepSimulation();
	virtual void renderScene();
	sim_data* get_results();
	

	void resetCamera()
	{
		float dist = 0.06*SCALE;
		float pitch = -20;
		float yaw = 52;

		float targetPos[3]={0,0,0};
		m_guiHelper->resetCamera(dist,yaw,pitch,targetPos[0],targetPos[1],targetPos[2]);
	}

	bool exitFlag;
	bool applyTestForce = false;
	bool releaseTestForce = false;
	Parameters* parameters =  new Parameters();

};

#endif //BASIC_DEMO_PHYSICS_SETUP_H
