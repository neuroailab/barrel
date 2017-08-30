

#include "Simulation_utility.hpp"

// Helper Functions for simulation
// ================================

btCollisionShape* createCylinderShape(btVector3 dimensions){
	btCollisionShape* colShape = new btCylinderShapeX(SCALE*dimensions);
	// std::cout << "-> cylinder shape created (scaled): " << SCALE*dimensions[0] << ", " << SCALE*dimensions[1] << ", "
	// 										<< SCALE*dimensions[2] << ", " << std::endl;
	return colShape;
}

btCollisionShape* createSphereShape(btScalar radius){
	btCollisionShape* colShape = new btSphereShape(SCALE*radius);
	// std::cout << "-> sphere shape created (scaled): " << SCALE*radius << std::endl;
	return colShape;

}

// function to greate frostum shape for whisker units
btConvexHullShape* createFrostumShapeX(btScalar height, btScalar r_base, btScalar r_top, int NUM_POINTS){
	btConvexHullShape* colShape = new btConvexHullShape();

	btScalar dalpha = 2*PI/NUM_POINTS; // calculate delta angle of points in base and top

	// add points of base
	float tmp_anlge = 0.;
	for(int i=0; i<NUM_POINTS; i++){
		btVector3 point = btVector3(-height*SCALE,cos(i*dalpha)*r_base*SCALE,sin(i*dalpha)*r_base*SCALE);
		colShape->addPoint(point);
	}

	// add points of top
	for(int i=0; i<NUM_POINTS; i++){
		btVector3 point = btVector3(height*SCALE,cos(i*dalpha)*r_top*SCALE,sin(i*dalpha)*r_top*SCALE);
		colShape->addPoint(point);
	}
	
	return colShape;
}

// function to create dynamic body
btRigidBody* createDynamicBody(float mass, const btTransform& startTransform, btCollisionShape* shape,  const btVector4& color)
{
	btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

	//rigidbody is dynamic if and only if mass is non zero, otherwise static
	bool isDynamic = (mass != 0.f);

	btVector3 localInertia(0, 0, 0);
	if (isDynamic){
		shape->calculateLocalInertia(mass, localInertia);
	}
	else{
		std::cout << "Warning: body mass is zero!" << std::endl;
	}

	//using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
	btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, myMotionState, shape, localInertia);

	btRigidBody* body = new btRigidBody(cInfo);
	// body->setContactProcessingThreshold(m_defaultContactProcessingThreshold);

	body->setUserIndex(-1);
	return body;
}

// function to create frame
btTransform createFrame(btVector3 origin, btVector3 rotation){
	btTransform frame;
	frame = btTransform::getIdentity();
	frame.setOrigin(SCALE*origin);
	frame.getBasis().setEulerZYX(rotation[0],rotation[1],rotation[2]);
	return frame;
}

// function to translate frame
void translateFrame(btTransform& transform, btVector3 origin){

	transform.setOrigin(SCALE*origin);
}

// function to rotate frame with eular angles
void rotateFrame(btTransform& transform, btVector3 rotation){

	btScalar rx = rotation[0];	// roll
	btScalar ry = rotation[1];	// pitch
	btScalar rz = rotation[2];	// yaw

	transform.getBasis().setEulerZYX(rx,ry,rz);
}

// function to convert a btVector3 to a float vector
std::vector<float> btVecToFloat(btVector3 btVec){
	std::vector<float> floatVec(3,0);
	floatVec[0] = float(btVec[0]);
	floatVec[1] = float(btVec[1]);
	floatVec[2] = float(btVec[2]);
	// std::cout << "floatVec size: " << floatVec.size() << std::endl;
	return floatVec;
}


