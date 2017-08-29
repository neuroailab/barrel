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


#include "Whisker.hpp"


Whisker::Whisker(btDiscreteDynamicsWorld* world, btAlignedObjectArray<btCollisionShape*>* shapes, Parameters* parameters, btRigidBody* ref){
		// save parameters and global variables to whisker object
		m_collisionShapes = shapes;	// shape vector pointer
		m_dynamicsWorld = world;	// simulation world pointer
		para = parameters;			// whisker parameters
		refBody = ref; 				// reference point for base point position

		m_angle = 0.;		// initialize protraction angle
		m_time = 0;			// initialize time
}

// function to create whisker
void Whisker::createWhisker(std::string w_name){

	std::cout << "Creating whisker " << w_name << "..." << std::endl;

	int NUM_UNITS = para->NUM_UNITS;	// set number of units
	int NUM_LINKS = NUM_UNITS - 1;		// set number of links
	
	config = get_parameters(w_name);	// get parameters for whisker configuration

	// std::cout << config.id << std::endl;
	// std::cout << config.row << std::endl;
	// std::cout << config.col << std::endl;
	// std::cout << config.L << std::endl;
	// std::cout << config.a << std::endl;
	// std::cout << config.base_pos[0] << ", " << config.base_pos[1] << ", " << config.base_pos[2] << std::endl;
	// std::cout << config.base_rot[0] << ", " << config.base_rot[1] << ", " << config.base_rot[2] << std::endl;


	std::cout << "Length: " << config.L << std::endl;

	// calculate individual whisker parameters
	float base_radius = calc_base_radius(config.row,config.col); // base radius
	std::cout << "R_base: " << base_radius << std::endl;

	float tip_radius = para->BT_RATIO*base_radius; // tip radius
	std::cout << "R_tip: " << tip_radius << std::endl;

	float taper = (base_radius-tip_radius)/NUM_UNITS; // slope of tapered whisker
	float unit_length = config.L / NUM_UNITS; // height or length of one unit

	// calculate curvature of the whisker
	std::vector<float> unit_radius = calc_unit_radius(NUM_UNITS, base_radius, tip_radius );
	std::vector<float> unit_mass = calc_mass(NUM_UNITS, unit_length,base_radius,tip_radius,para->DENSITY);
	std::vector<float> link_angles = get_angles_from_curvature(config.L, config.a, NUM_UNITS);
	std::vector<float> unit_youngs = calc_young_modulus(NUM_UNITS, para->E_BASE, para->E_TIP);
	std::vector<float> unit_inertias = calc_inertia(NUM_UNITS, base_radius, tip_radius);
	std::vector<float> link_stiffness = calc_stiffness(NUM_LINKS, unit_youngs, unit_inertias, unit_length);
	std::vector<float> link_damping = calc_damping(NUM_LINKS, link_stiffness, unit_mass, unit_length/2, para->ZETA_BASE, para->ZETA_TIP);


	/// create whisker
	/// ====================================

	btTransform baseTransform = createFrame(config.base_pos);
	btCollisionShape* baseShape = createSphereShape(para->BLOW*unit_radius[0]);
	m_collisionShapes->push_back(baseShape);
	
	// add whisker base and world
	base = createDynamicBody(0.1,baseTransform,baseShape);
	m_dynamicsWorld->addRigidBody(base);

	// set up constraint for whole whisker to head
	// ===========================================================
	btTransform baseFrame = createFrame();	
	btTransform refFrame = createFrame();
	btGeneric6DofConstraint* fixedConstraint = new btGeneric6DofConstraint(*refBody, *base, refFrame, baseFrame, false);

	// make whisker base fixed
	if(para->TEST || para->OPT){
		fixedConstraint->setLinearLowerLimit(btVector3(0,0,0));
		fixedConstraint->setLinearUpperLimit(btVector3(0,0,0));

	}
	else{
		fixedConstraint->setLinearLowerLimit(config.base_pos*SCALE);
		fixedConstraint->setLinearUpperLimit(config.base_pos*SCALE);

	}
	
	// set rotation in respect to origin
		fixedConstraint->setAngularLowerLimit(btVector3(0,0,0));
		fixedConstraint->setAngularUpperLimit(btVector3(0,0,0));

	// add constraint to world
	m_dynamicsWorld->addConstraint(fixedConstraint,true);
	fixedConstraint->setDbgDrawSize(btScalar(5.f));

	// set up constraint for whisker base and first unit
	// ===========================================================
	
	// set whisker orientation
	btVector3 orientation = btVector3(config.base_rot[2],config.base_rot[1],config.base_rot[0]);
	rotateFrame(baseTransform,orientation);

	// calculate distance between unit COMs
	float unit_offset = unit_length/2;

	// set unit mass
	btScalar  mass(unit_mass[0]);
	std::cout << unit_radius[0] << std::endl;
	
	// generate shape for unit
	btConvexHullShape* unitShape = createFrostumShapeX(unit_length/2, para->BLOW*unit_radius[0], para->BLOW*unit_radius[1], 12);
	unitShape->setSafeMargin(unit_radius[1]);
	m_collisionShapes->push_back(unitShape);

	if(para->NO_CURVATURE){ // set all angles between units to zero
		link_angles[0]=0.;
	}

	// set position and rotation of current unit
	btTransform unitTransform = createFrame(btVector3(unit_offset,0.,0.));
	
	// add unit to whisker and world
	btRigidBody* unit_first = createDynamicBody(mass,baseTransform*unitTransform,unitShape);
	whisker.push_back(unit_first);	
	m_dynamicsWorld->addRigidBody(unit_first);

	// make units active
	base->setActivationState(DISABLE_DEACTIVATION);
	unit_first->setActivationState(DISABLE_DEACTIVATION);

	// initialize transforms and set frames at end of frostum
	btTransform unitFrame = createFrame(btVector3(-unit_length/2,0,0),btVector3(0,0,0));
		

	// create link (between units) constraint
	baseConstraint = new btGeneric6DofSpring2Constraint(*base, *unit_first, baseFrame, unitFrame);


	// make whisker base fixed
	baseConstraint->setLinearLowerLimit(btVector3(0,0,0));
	baseConstraint->setLinearUpperLimit(btVector3(0,0,0));

	if(para->TEST || para->OPT){
		// set rotation of whisker base to zero
		baseConstraint->setAngularLowerLimit(btVector3(0,0,0));
		baseConstraint->setAngularUpperLimit(btVector3(0,0,0));
	}
	else{
		// set rotation about z axis
		baseConstraint->setAngularLowerLimit(-orientation);
		baseConstraint->setAngularUpperLimit(-orientation);
	}
	
	
	// add constraint to world
	m_dynamicsWorld->addConstraint(baseConstraint,true);
	baseConstraint->setDbgDrawSize(btScalar(5.f));
 
	// enable feedback (mechanical response)
	baseConstraint->setJointFeedback(&baseFeedback);

	// add remaining units
	for(int i=1;i<NUM_UNITS;++i) {

		// compute COM offset for next unit
		unit_offset = unit_offset + unit_length;

		// set unit mass
		btScalar  mass(unit_mass[i]);
		// btScalar  mass(0);
		std::cout << unit_radius[i] << std::endl;
		
		// generate shape for unit
		btConvexHullShape* unitShape = createFrostumShapeX(unit_length/2, para->BLOW*unit_radius[i], para->BLOW*unit_radius[i+1], 12);
		unitShape->setSafeMargin(unit_radius[i+1]);
		m_collisionShapes->push_back(unitShape);

		if(para->NO_CURVATURE){ // set all angles between units to zero
			link_angles[i]=0.;
		}

		// set position and rotation of current unit
		btTransform unitTransform = createFrame(btVector3(unit_offset,0.,0.));
		
		// add unit to whisker and world
		btRigidBody* unit_curr = createDynamicBody(mass,baseTransform*unitTransform,unitShape);
		whisker.push_back(unit_curr);	
		m_dynamicsWorld->addRigidBody(unit_curr);	 

		// get previous unit
		btRigidBody* unit_prev = whisker[i-1];   // get unit i

		// make units active
		// unit_prev->setActivationState(DISABLE_DEACTIVATION);
		unit_curr->setActivationState(DISABLE_DEACTIVATION);

		// create frames for constraint
		btTransform frameInPrev;				
		btTransform frameInCurr;

		// initialize transforms of both units and set frames at end of frostums
		frameInPrev = createFrame(btVector3(unit_length/2,0,0),btVector3(0,0,0));				
		frameInCurr = createFrame(btVector3(-unit_length/2,0,0),btVector3(0,0,0));
				
		// create link (between units) constraint
		btGeneric6DofSpring2Constraint* link = new btGeneric6DofSpring2Constraint(*unit_prev, *unit_curr, frameInPrev,frameInCurr);
		// links.push_back(link); // save constraint


		// set spring parameters of links - are not physical measures
		// ----------------------------------------------------------

		if(para->NO_CURVATURE){
			link_angles[i]=0.0;
		}
		
		link->setLinearLowerLimit(btVector3(0., 0.0, 0.0)); // lock the units
		link->setLinearUpperLimit(btVector3(0., 0.0, 0.0));

		link->setAngularLowerLimit(btVector3(0.,1.,1.)); // lock angles between units at x axis but free around y and z axis
		link->setAngularUpperLimit(btVector3(0.,0.,0.));

		// add constraint to world
		m_dynamicsWorld->addConstraint(link, true); // true -> collision between linked bodies disabled
		link->setDbgDrawSize(btScalar(5.f));

		// enable springs at constraints between units
		std::cout << "stiffness x: " << para->stiffness_x[i] << std::endl;
		std::cout << "damping x: " << para->damping_x[i] << std::endl;
		link->enableSpring(3,true);
		link->setStiffness(3,para->stiffness_x[i],false);
		link->setDamping(3,para->damping_x[i],false);
		link->setEquilibriumPoint(3,0.);

		std::cout << "stiffness y: " << para->stiffness_y[i] << std::endl;
		std::cout << "damping y: " << para->damping_y[i] << std::endl;
		link->enableSpring(4,true);
		link->setStiffness(4,para->stiffness_y[i],false);
		link->setDamping(4,para->damping_y[i],false);
		link->setEquilibriumPoint(4,0.);

		std::cout << "stiffness z: " << para->stiffness_z[i] << std::endl;
		std::cout << "damping z: " << para->damping_z[i] << std::endl;
		link->enableSpring(5,true);
		link->setStiffness(5,para->stiffness_z[i],false);
		link->setDamping(5,para->damping_z[i],false);
		link->setEquilibriumPoint(5,-link_angles[i]);
		
		
		// link->setParam(BT_CONSTRAINT_CFM, 0.000001, 3);
		// link->setParam(BT_CONSTRAINT_CFM, 0.000001 ,4);
		// link->setParam(BT_CONSTRAINT_CFM, 0.000001 ,5);








		

	} 
	 
	
	// // set up constraints for whisker links (between units)
	// // ==============================================================

	// for(int i=NUM_LINKS;i>0;--i) {

	// 	// get neighboring units
	// 	btRigidBody* unitA = whisker[i];   // get unit i
	// 	btRigidBody* unitB = whisker[i-1]; // get unit i+1

	// 	// make units active
	// 	unitA->setActivationState(DISABLE_DEACTIVATION);
	// 	unitB->setActivationState(DISABLE_DEACTIVATION);

	// 	// create frames for constraint
	// 	btTransform frameInA;				
	// 	btTransform frameInB;

	// 	// initialize transforms of both units and set frames at end of frostums
	// 	frameInA = createFrame(btVector3(-unit_length/2,0,0),btVector3(0,0,0));				
	// 	frameInB = createFrame(btVector3(unit_length/2,0,0),btVector3(0,0,0));
				
	// 	// create link (between units) constraint
	// 	btGeneric6DofSpring2Constraint* link = new btGeneric6DofSpring2Constraint(*unitA, *unitB, frameInA,frameInB);
	// 	// links.push_back(link); // save constraint


	// 	// set spring parameters of links - are not physical measures
	// 	// ----------------------------------------------------------

	// 	if(para->NO_CURVATURE){
	// 		link_angles[i]=0.0;
	// 	}
		
	// 	link->setLinearLowerLimit(btVector3(0., 0.0, 0.0)); // lock the units
	// 	link->setLinearUpperLimit(btVector3(0., 0.0, 0.0));

	// 	link->setAngularLowerLimit(btVector3(0.,1.,1.)); // lock angles between units at x axis but free around y and z axis
	// 	link->setAngularUpperLimit(btVector3(0.,0.,0.));

	// 	// add constraint to world
	// 	m_dynamicsWorld->addConstraint(link, true); // true -> collision between linked bodies disabled
	// 	link->setDbgDrawSize(btScalar(5.f));

	// 	// enable springs at constraints between units
	// 	std::cout << "stiffness x: " << para->stiffness_x[i] << std::endl;
	// 	std::cout << "damping x: " << para->damping_x[i] << std::endl;
	// 	link->enableSpring(3,true);
	// 	link->setStiffness(3,para->stiffness_x[i],false);
	// 	link->setDamping(3,para->damping_x[i],false);
	// 	link->setEquilibriumPoint(3,0.);

	// 	std::cout << "stiffness y: " << para->stiffness_y[i] << std::endl;
	// 	std::cout << "damping y: " << para->damping_y[i] << std::endl;
	// 	link->enableSpring(4,true);
	// 	link->setStiffness(4,para->stiffness_y[i],false);
	// 	link->setDamping(4,para->damping_y[i],false);
	// 	link->setEquilibriumPoint(4,0.);

	// 	std::cout << "stiffness z: " << para->stiffness_z[i] << std::endl;
	// 	std::cout << "damping z: " << para->damping_z[i] << std::endl;
	// 	link->enableSpring(5,true);
	// 	link->setStiffness(5,para->stiffness_z[i],false);
	// 	link->setDamping(5,para->damping_z[i],false);
	// 	link->setEquilibriumPoint(5,link_angles[i]);
		
		
	// 	// link->setParam(BT_CONSTRAINT_CFM, 0.000001, 3);
	// 	// link->setParam(BT_CONSTRAINT_CFM, 0.000001 ,4);
	// 	// link->setParam(BT_CONSTRAINT_CFM, 0.000001 ,5);

		
	// }

	// // set up constraint for whisker base
	// // ===========================================================

	// // make units active
	// base->setActivationState(DISABLE_DEACTIVATION);

	// // initialize transforms of both units and set frames at end of frostum
	// btTransform baseFrame = createFrame();				
	// btTransform whiskerFrame = createFrame(btVector3(-unit_length/2,0,0),btVector3(0,0,0));
		

	// // create link (between units) constraint
	// baseConstraint = new btGeneric6DofSpring2Constraint(*base, *whisker[0], baseFrame, whiskerFrame);


	// // make whisker base fixed
	// baseConstraint->setLinearLowerLimit(btVector3(0,0,0));
	// baseConstraint->setLinearUpperLimit(btVector3(0,0,0));

	// if(para->TEST || para->OPT){
	// 	// set rotation of whisker base to zero
	// 	baseConstraint->setAngularLowerLimit(btVector3(0,0,0));
	// 	baseConstraint->setAngularUpperLimit(btVector3(0,0,0));
	// }
	// else{
	// 	// set rotation about z axis
	// 	btVector3 joint_rot = btVector3(-config.base_rot[2],-config.base_rot[1],-config.base_rot[0]);
	// 	// btVector3 joint_rot = btVector3(0,0,0);
	// 	baseConstraint->setAngularLowerLimit(joint_rot);
	// 	baseConstraint->setAngularUpperLimit(joint_rot);
	// }
	
	
	// // add constraint to world
	// m_dynamicsWorld->addConstraint(baseConstraint,true);
	// baseConstraint->setDbgDrawSize(btScalar(5.f));
 
	// // enable feedback (mechanical response)
	// baseConstraint->setJointFeedback(&baseFeedback);


	// // set up constraint for whole whisker to head
	// // ===========================================================
	// btTransform refFrame = createFrame();
	// btGeneric6DofConstraint* fixedConstraint = new btGeneric6DofConstraint(*refBody, *base, refFrame, baseFrame, false);

	// // make whisker base fixed
	// if(para->TEST || para->OPT){
	// 	fixedConstraint->setLinearLowerLimit(btVector3(0,0,0));
	// 	fixedConstraint->setLinearUpperLimit(btVector3(0,0,0));

	// }
	// else{
	// 	fixedConstraint->setLinearLowerLimit(config.base_pos*SCALE);
	// 	fixedConstraint->setLinearUpperLimit(config.base_pos*SCALE);

	// }
	
	// // set rotation in respect to origin
	// 	fixedConstraint->setAngularLowerLimit(btVector3(0,0,0));
	// 	fixedConstraint->setAngularUpperLimit(btVector3(0,0,0));

	// // add constraint to world
	// m_dynamicsWorld->addConstraint(fixedConstraint,true);
	// fixedConstraint->setDbgDrawSize(btScalar(5.f));

	
	std::cout << "done." << std::endl;
}


btRigidBody* Whisker::get_unit(int idx){
	return whisker[idx];
}


// function to rotate whisker about base point axis
void Whisker::moveWhisker(float time, float freq, float angle_max){

	// calculate target position and velocity
	float theta = angle_max*(sin(2*PI*freq*time));
	float theta_dt = angle_max*cos(2*PI*freq*time)*(2*PI*freq);

	btVector3 angularPosition;
	btVector3 angularVelocity;

	// if in TEST or OPT mode no torsion
	if(!(para->TEST || para->OPT)){
		float phi_dt = theta_dt * get_dphi(config.row);
		float zeta_dt = theta_dt * get_dzeta(config.row);	

		float phi = theta * get_dphi(config.row);
		float zeta = theta * get_dzeta(config.row);

		angularPosition = btVector3(zeta, phi, theta);
		angularVelocity = btVector3(zeta_dt, phi_dt, theta_dt);
	}
	else{
		angularPosition = btVector3(0, 0, theta);
		angularVelocity = btVector3(0, 0, theta_dt);
	}

	// update angular position in constraint
	baseConstraint->getRotationalLimitMotor(0)->m_loLimit = angularPosition[0];
	baseConstraint->getRotationalLimitMotor(1)->m_loLimit = angularPosition[1];
	baseConstraint->getRotationalLimitMotor(2)->m_loLimit = angularPosition[2];

	baseConstraint->getRotationalLimitMotor(0)->m_hiLimit = angularPosition[0];
	baseConstraint->getRotationalLimitMotor(1)->m_hiLimit = angularPosition[1];
	baseConstraint->getRotationalLimitMotor(2)->m_hiLimit = angularPosition[2];

	
}

// function to get torque at whisker base
std::vector<float> Whisker::getBPTorques(){

	std::vector<float> torques = btVecToFloat(baseConstraint->getJointFeedback()->m_appliedTorqueBodyA);
	if(para->PRINT){
		std::cout << "Mx : " << torques[0] << std::endl;
		std::cout << "My : " << torques[1] << std::endl;
		std::cout << "Mz : " << torques[2] << std::endl;
	}
	
	return torques;

}

// function to get forces at whisker base
std::vector<float> Whisker::getBPForces(){

	std::vector<float> forces = btVecToFloat(baseConstraint->getJointFeedback()->m_appliedForceBodyA);
	if(para->PRINT){
		std::cout << "Fx : " << forces[0] << std::endl;
		std::cout << "Fy : " << forces[1] << std::endl;
		std::cout << "Fz : " << forces[2] << std::endl;
	}
	
	return forces;

}

// function to obtain the world coordinates of each whisker unit
std::vector<float> Whisker::getX(){

	std::vector<float> trajectories;
	trajectories.push_back(base->getCenterOfMassTransform().getOrigin()[0]);

	// loop through units and get world coordinates of each
	for (int i=0; i<whisker.size(); i++){
		float x = whisker[i]->getCenterOfMassTransform().getOrigin()[0];
		if(para->PRINT){
			std::cout << "x " << i << ": " << x << std::endl;
		}
		
		trajectories.push_back(x);
	}

	return trajectories;

}

// function to obtain the world coordinates of each whisker unit
std::vector<float> Whisker::getY(){

	std::vector<float> trajectories;
	trajectories.push_back(base->getCenterOfMassTransform().getOrigin()[1]);

	// loop through units and get world coordinates of each
	for (int i=0; i<whisker.size(); i++){
		float y = whisker[i]->getCenterOfMassTransform().getOrigin()[1];
		if(para->PRINT){
			std::cout << "y " << i << ": " << y << std::endl;
		}
		trajectories.push_back(y);
	}

	return trajectories;

}

// function to obtain the world coordinates of each whisker unit
std::vector<float> Whisker::getZ(){

	std::vector<float> trajectories;
	trajectories.push_back(base->getCenterOfMassTransform().getOrigin()[2]);

	// loop through units and get world coordinates of each
	for (int i=0; i<whisker.size(); i++){
		float z = whisker[i]->getCenterOfMassTransform().getOrigin()[2];
		if(para->PRINT){
			std::cout << "z " << i << ": " << z << std::endl;
		}
		trajectories.push_back(z);
	}

	return trajectories;

}
