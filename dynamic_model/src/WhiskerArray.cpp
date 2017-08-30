

#include "WhiskerArray.hpp"



WhiskerArray::WhiskerArray(btDiscreteDynamicsWorld* world, btAlignedObjectArray<btCollisionShape*>* shapes, Parameters* parameters){
	
	// add head/array center 
	btTransform centerTransform = createFrame();
	btEmptyShape* centerShape = new btEmptyShape();
	shapes->push_back(centerShape);
	
	center = createDynamicBody(0,centerTransform,centerShape);
	world->addRigidBody(center);

	// create Whiskers
	for(int w=0;w<parameters->WHISKER_NAMES.size();w++){
		Whisker* whisker = new Whisker(world, shapes,parameters, center);
		whisker->createWhisker(parameters->WHISKER_NAMES[w]);
		m_whiskerArray.push_back(whisker);

	}
		
}

btAlignedObjectArray<Whisker*> WhiskerArray::getArray(){
	return m_whiskerArray;
}

void WhiskerArray::applyForceToWhisker(int idx_whisker, int idx_unit, btVector3 force){

	// m_whiskerArray[idx_whisker]->get_unit(idx_unit)->applyForce(force,btVector3(0,0,0));
	m_whiskerArray[idx_whisker]->get_unit(idx_unit)->setGravity(force + m_whiskerArray[idx_whisker]->get_unit(idx_unit)->getGravity());
	// std::cout << "Force applied: " << m_whiskerArray[idx_whisker]->get_unit(idx_unit)->getGravity()[2] << std::endl;
}

void WhiskerArray::releaseForceToWhisker(int idx_whisker, int idx_unit, btVector3 gravity){

	// m_whiskerArray[idx_whisker]->get_unit(idx_unit)->applyForce(force,btVector3(0,0,0));
	std::cout << "Force applied: " << m_whiskerArray[idx_whisker]->get_unit(idx_unit)->getGravity()[2] << std::endl;
	m_whiskerArray[idx_whisker]->get_unit(idx_unit)->setGravity(gravity);
	std::cout << "Force released: " << m_whiskerArray[idx_whisker]->get_unit(idx_unit)->getGravity()[2] << std::endl;

}

void WhiskerArray::moveArray(float time, float freq, float angle_max){

	for (int i=0;i<m_whiskerArray.size();i++){
		m_whiskerArray[i]->moveWhisker(time, freq, angle_max);
	}

}

// dummy function for future head translation and rotation
void WhiskerArray::moveCenter(float time, float freq, float pos_max){

	// calculate target position and velocity
	float y = pos_max*(sin(2*PI*freq*time));

	btTransform trans;
	trans = center->getCenterOfMassTransform();
	btMotionState* motionState = center->getMotionState();

	btVector3 position = center->getCenterOfMassPosition();

	position[1] = y;
    trans.setOrigin(position);

    center->setCenterOfMassTransform(trans);


    motionState->setWorldTransform(trans);

}

// function to retrieve torques at base points
std::vector<std::vector<float>> WhiskerArray::getAllBPTorques(){

	std::vector<std::vector<float>> all_bp_torques;
	all_bp_torques.reserve(m_whiskerArray.size());

	for(int w=0; w < m_whiskerArray.size(); w++){
		all_bp_torques.push_back(m_whiskerArray[w]->getBPTorques());

	}
	// std::cout << "all torque size: " << all_bp_torques[0].size() << std::endl;
	return all_bp_torques;
}

// function to retrieve forces at base points
std::vector<std::vector<float>> WhiskerArray::getAllBPForces(){

	std::vector<std::vector<float>> all_bp_forces;
	all_bp_forces.reserve(m_whiskerArray.size());

	for(int w=0; w < m_whiskerArray.size(); w++){
		all_bp_forces.push_back(m_whiskerArray[w]->getBPForces());

	}
	// std::cout << "all force size: " << all_bp_forces[0].size() << std::endl;
	return all_bp_forces;
}

// function to obtain x coordinates of all whisker units
std::vector<std::vector<float>> WhiskerArray::getAllX(){

	std::vector<std::vector<float>> all_x;
	all_x.reserve(m_whiskerArray.size());

	for(int w=0; w < m_whiskerArray.size(); w++){
		all_x.push_back(m_whiskerArray[w]->getX());

	}
	// std::cout << "all force size: " << all_bp_forces[0].size() << std::endl;
	return all_x;
}

// function to obtain y coordinates of all whisker units
std::vector<std::vector<float>> WhiskerArray::getAllY(){

	std::vector<std::vector<float>> all_y;
	all_y.reserve(m_whiskerArray.size());

	for(int w=0; w < m_whiskerArray.size(); w++){
		all_y.push_back(m_whiskerArray[w]->getY());

	}
	// std::cout << "all force size: " << all_bp_forces[0].size() << std::endl;
	return all_y;
}

// function to obtain z coordinates of all whisker units
std::vector<std::vector<float>> WhiskerArray::getAllZ(){

	std::vector<std::vector<float>> all_z;
	all_z.reserve(m_whiskerArray.size());

	for(int w=0; w < m_whiskerArray.size(); w++){
		all_z.push_back(m_whiskerArray[w]->getZ());

	}
	// std::cout << "all force size: " << all_bp_forces[0].size() << std::endl;
	return all_z;
}