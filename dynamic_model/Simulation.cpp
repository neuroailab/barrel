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


void Simulation::stepSimulation(){

	// enable force at first step for optimization
	if((parameters->TEST_FORCE!=0) && m_time == 0.){
		applyTestForce = true;
		releaseTestForce = false;
	}

	m_time += parameters->TIME_STEP; // increase time
	
	// apply force to whisker at first step if force not zero
	if((parameters->TEST_FORCE!=0) && applyTestForce){
			whiskerArray->applyForceToWhisker(0, 3, btVector3(0,0,parameters->TEST_FORCE));
			applyTestForce = false;
			releaseTestForce = true;
		}

	// release applied force if stop time exceeded
	if((m_time > parameters->TIME_STOP/2) && releaseTestForce){
		whiskerArray->releaseForceToWhisker(0, 3, gravity);
		releaseTestForce = false;
	}

	// run simulation as long as stop time not exceeded
	if(m_time < parameters->TIME_STOP){

		// move array if in ACTIVE mode
		if(parameters->ACTIVE){
			
			whiskerArray->moveArray(m_time, parameters->WHISK_FREQ, parameters->WHISK_AMP); // move array with defined frequency and amplitude
			
		}
		
		// whiskerArray->moveCenter(m_time, 0.5, 7);

		
		// step simulation
		m_dynamicsWorld->stepSimulation(parameters->TIME_STEP,parameters->NUM_STEP_INT,parameters->TIME_STEP/parameters->NUM_STEP_INT);

		// read mechanical responses and dump data to vector
		data_dump->T.push_back(m_time);
		data_dump->M.push_back(whiskerArray->getAllBPTorques());
		data_dump->F.push_back(whiskerArray->getAllBPForces());
		data_dump->X.push_back(whiskerArray->getAllX());
		data_dump->Y.push_back(whiskerArray->getAllY());
		data_dump->Z.push_back(whiskerArray->getAllZ());

		// print current time if enabled
		if(parameters->PRINT){
			std::cout << "Time elapsed: " << m_time << std::endl;
		}

		// draw debug if enabled
	    if(parameters->DEBUG){
	    	m_dynamicsWorld->debugDrawWorld();
	    }

	    // set exit flag to zero
	    exitFlag = 0;

	}
	else{
		
		// timeout -> set exit flg
		exitFlag = 1;
		
	}
	
    
    // CommonRigidBodyBase::stepSimulation(dt);
}

void Simulation::initPhysics()
{	
	// set visual axis
	m_guiHelper->setUpAxis(2);

	// create empty dynamics world
	m_collisionConfiguration = new btDefaultCollisionConfiguration(); 
    m_dispatcher = new	btCollisionDispatcher(m_collisionConfiguration); 

    // broadphase algorithm
    m_broadphase = new btDbvtBroadphase();
    // m_broadphase = new btAxisSweep3(worldMin,worldMax);

   
	// select solver
	switch(parameters->SOLVER){

		case 0 : {	
			std::cout << "Using btSequentialImpulseConstraintSolver..." << std::endl;
			m_solver = new btSequentialImpulseConstraintSolver();

			m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher,m_broadphase,m_solver,m_collisionConfiguration);
			// m_dynamicsWorld ->getSolverInfo().m_minimumSolverBatchSize = 128;
			break;
		}
		case 1 : {	
			std::cout << "Using btNNCGConstraintSolver..." << std::endl;
			m_solver = new btNNCGConstraintSolver();

			m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher,m_broadphase,m_solver,m_collisionConfiguration);
			m_dynamicsWorld ->getSolverInfo().m_minimumSolverBatchSize = 128;
			break;
		}
		case 2 : {	
			std::cout << "Using btDantzigSolver (MLCP)..." << std::endl;
			btDantzigSolver* mlcp = new btDantzigSolver();		
			btMLCPSolver* sol = new btMLCPSolver(mlcp);
			m_solver = sol;

			m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher,m_broadphase,m_solver,m_collisionConfiguration);
			m_dynamicsWorld ->getSolverInfo().m_minimumSolverBatchSize = 1;
			break;
		}
		case 3 : {
			std::cout << "Using btSolveProjectedGaussSeidel (MLCP)..." << std::endl;
			btSolveProjectedGaussSeidel* mlcp = new btSolveProjectedGaussSeidel;
		
			btMLCPSolver* sol = new btMLCPSolver(mlcp);
			m_solver = sol;
			m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher,m_broadphase,m_solver,m_collisionConfiguration);
			m_dynamicsWorld ->getSolverInfo().m_minimumSolverBatchSize = 1;
			break;
		}
		default : {
			std::cout << "Using btSequentialImpulseConstraintSolver..." << std::endl;
			m_solver = new btSequentialImpulseConstraintSolver();

			m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher,m_broadphase,m_solver,m_collisionConfiguration);
			// m_dynamicsWorld ->getSolverInfo().m_minimumSolverBatchSize = 128;
			break;
		}
	}

	// set number of iterations
	m_dynamicsWorld->getSolverInfo().m_numIterations = 100;

	// set gravity
	m_dynamicsWorld->setGravity(gravity);

    // create debug drawer
	m_guiHelper->createPhysicsDebugDrawer(m_dynamicsWorld);

	if (m_dynamicsWorld->getDebugDrawer())
		m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_DrawConstraints+btIDebugDraw::DBG_DrawContactPoints+btIDebugDraw::DBG_DrawWireframe);
			// btIDebugDraw::DBG_DrawConstraintLimits); //btIDebugDraw::DBG_DrawConstraints++btIDebugDraw::DBG_DrawConstraintLimits btIDebugDraw::DBG_DrawWireframe+

	// create ground plane at origin
	btCollisionShape* groundShape = new btStaticPlaneShape(SCALE*btVector3(0,0,1),SCALE*40);
	m_collisionShapes.push_back(groundShape);

	btTransform groundTransform;
	groundTransform.setIdentity();
	translateFrame(groundTransform,btVector3(0,0,-50.));
	btScalar mass(0.);
	btRigidBody* body = createRigidBody(mass,groundTransform,groundShape, btVector4(0,0,1,1));


	// create whisker array
	whiskerArray = new WhiskerArray(m_dynamicsWorld, &m_collisionShapes, parameters);
		
	// collision mode 1: stationary peg
	if(parameters->COLLIDE==1){
		// create object to collide with
		btCollisionShape* peg = new btCylinderShapeZ(SCALE*btVector3(0.001,0.001,0.02));
		m_collisionShapes.push_back(peg);

		btTransform pegTransform;
		pegTransform.setIdentity();
		translateFrame(pegTransform,btVector3(0.025,-0.02,0));
		btScalar mass_peg(0.);
		btRigidBody* peg_body = createRigidBody(mass_peg,pegTransform,peg, btVector4(0,0,1,1));
	}
	// collision mode 2: falling peg
	else if(parameters->COLLIDE==2){
		// create object to collide with
		btCollisionShape* peg = new btCylinderShapeZ(SCALE*btVector3(0.0005,0.0005,0.02));
		m_collisionShapes.push_back(peg);

		btTransform pegTransform = createFrame(btVector3(0.015,0.,0.03),btVector3(PI/2,0,0));
		btScalar mass_peg(1.);
		btRigidBody* peg_body = createRigidBody(mass_peg,pegTransform,peg, btVector4(0,0,1,1));
	}
	

	// initialize simulation time
	m_time = 0;

	// generate graphics
	m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);
}


void Simulation::exitPhysics(){

	removePickingConstraint();
	std::cout << "- Picking constraints removed." << std::endl;

	//remove the rigidbodies from the dynamics world and delete them	
	if (m_dynamicsWorld)
	{

        for (int i = m_dynamicsWorld->getNumConstraints() - 1; i >= 0; i--)
        {
            m_dynamicsWorld->removeConstraint(m_dynamicsWorld->getConstraint(i));
            
        }
        std::cout << "- Constraints removed." << std::endl;

		for (int i = m_dynamicsWorld->getNumCollisionObjects() - 1; i >= 0; i--)
		{
			btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[i];
			btRigidBody* body = btRigidBody::upcast(obj);
			if (body && body->getMotionState())
			{
				delete body->getMotionState();
				// std::cout << "- Body motion state " << i << " removed." << std::endl;
			}
			m_dynamicsWorld->removeCollisionObject(obj);
			delete obj;
			// std::cout << "- Collision objects " << i << " removed." << std::endl;
		}
		std::cout << "- Bodies and motion states removed." << std::endl;
	}

	//delete collision shapes
	for (int j = 0; j<m_collisionShapes.size(); j++)
	{
		btCollisionShape* shape = m_collisionShapes[j];
		delete shape;
		// std::cout << "- Collision shapes " << j << " removed." << std::endl;
	}
	m_collisionShapes.clear();
	std::cout << "- Collision shapes removed." << std::endl;

	delete m_dynamicsWorld;
	m_dynamicsWorld=0;
	std::cout << "- Dynamic World removed." << std::endl;

	delete m_solver;
	m_solver=0;
	std::cout << "- Solver removed." << std::endl;

	delete m_broadphase;
	m_broadphase=0;
	std::cout << "- Broadphase removed." << std::endl;

	delete m_dispatcher;
	m_dispatcher=0;
	std::cout << "- Dispatcher removed." << std::endl;

	delete m_collisionConfiguration;
	m_collisionConfiguration=0;
	std::cout << "- Collision Configuration removed." << std::endl;

	std::cout << "- Done." << std::endl;
}

sim_data* Simulation::get_results(){
	return data_dump;
}

void Simulation::renderScene()
{
	CommonRigidBodyBase::renderScene();
	
}


Simulation* SimulationCreateFunc(CommonExampleOptions& options)
{
	return new Simulation(options.m_guiHelper);

}


B3_STANDALONE_EXAMPLE(SimulationCreateFunc)



