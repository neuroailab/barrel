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


#include "Whiskers.h"

#include "btBulletDynamicsCommon.h"

#define PI 3.1415927

#include "LinearMath/btVector3.h"
#include "LinearMath/btAlignedObjectArray.h"
#include <iostream>

#include "CommonInterfaces/CommonRigidBodyBase.h"
#include "CommonInterfaces/CommonParameterInterface.h"

int collisionFilterGroup = int(btBroadphaseProxy::CharacterFilter);
int collisionFilterMask = int(btBroadphaseProxy::AllFilter ^ (btBroadphaseProxy::CharacterFilter));

struct Whiskers : public CommonRigidBodyBase
{
	Whiskers(struct GUIHelperInterface* helper)
		:CommonRigidBodyBase(helper)
	{
	}
	virtual ~Whiskers(){}
	virtual void initPhysics();
	virtual void stepSimulation(float dt);
	virtual void renderScene();

	bool m_once;
	btAlignedObjectArray<btJointFeedback*> m_jointFeedback;

	void resetCamera()
	{
		float dist = 2;
		float pitch = -20;
		float yaw = 52;
		float targetPos[3]={0,0,0};
		m_guiHelper->resetCamera(dist,yaw,pitch,targetPos[0],targetPos[1],targetPos[2]);
	}

	void createWhisker(float whiskerLength, float wradius, int nUnit, float wmass);
	// void setWhiskerMotion(btScalar deltaTime, btHingeConstraint* baseLink);
};

void Whiskers::stepSimulation(float dt){
    if (m_once)//m_once)
    {
        m_once=false;
        btGeneric6DofConstraint* baseConstraint = (btGeneric6DofConstraint*)m_dynamicsWorld->getConstraint(0);
        
        btRigidBody& base = baseConstraint->getRigidBodyA();
        btTransform trBase = base.getWorldTransform();
        btVector3 contraintAxisInWorld = trBase.getBasis()*baseConstraint->getFrameOffsetA().getBasis().getColumn(2);

        baseConstraint->getRigidBodyA().applyTorque(contraintAxisInWorld*10);
     //    btScalar torque = 10;
     //    btRigidBody* body = btRigidBody::upcast(m_dynamicsWorld->getCollisionObjectArray()[0]);
     //    btVector3 torqueVector;
     //    torqueVector = (body->getInvInertiaTensorWorld().inverse()*(body->getWorldTransform().getBasis()*torque)).getColumn(2);

    	// body->applyTorque(torqueVector*10);
    }

    m_dynamicsWorld->stepSimulation(dt);
    // CommonRigidBodyBase::stepSimulation(dt);
}

void Whiskers::initPhysics()
{	
	// m_Time = 0;
	// m_fCyclePeriod = 2000.f;

	float whiskerLength = 1;
	float wradius = 0.05;
	float wmass = 0.1;
	int nUnit = 13;


	m_guiHelper->setUpAxis(1);

	createEmptyDynamicsWorld();
	m_dynamicsWorld->setGravity(btVector3(0,-10,0));
	m_guiHelper->createPhysicsDebugDrawer(m_dynamicsWorld);

	if (m_dynamicsWorld->getDebugDrawer())
		m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_DrawWireframe+btIDebugDraw::DBG_DrawContactPoints);

	// create ground plane at origin
	btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0,1,0),50);
	m_collisionShapes.push_back(groundShape);

	btTransform groundTransform;
	groundTransform.setIdentity();
	groundTransform.setOrigin(btVector3(0,-50,0));

	{
		btScalar mass(0.);
		createRigidBody(mass,groundTransform,groundShape, btVector4(0,0,1,1));
	}


	{
		//create a few dynamic rigidbodies
		
		createWhisker(whiskerLength,wradius,nUnit,wmass);
		
	}

	
	m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);
	m_once = true;
}

void Whiskers::createWhisker(float whiskerLength, float rBase, int nUnit, float wmass){

	float rho = 1300.; // density
	float rTip = 0.02*rBase; // tip radius
	float taper = (rBase-rTip)/nUnit; // slope of tapered whisker

	float hUnit = whiskerLength / nUnit; // height or length of one unit
	float springLength = 0.1; // length of springs between units

	float bp_posz = 1; // z-position of basepoint

	float mUnit; // mass of one unit

	btTransform basePointTransform;
	basePointTransform.setIdentity();
	basePointTransform.setOrigin(btVector3(
								btScalar(0),
								btScalar(1),
								btScalar(0)));
	/// create whisker
	/// ====================================

	

	// initialize array of units that form one whisker
	btAlignedObjectArray<btRigidBody*> whisker;

	// set up geometry of whisker
	for(int i=0;i<nUnit;++i) {

		// initialize transform for whisker unit
		btTransform unitTransform;
		unitTransform.setIdentity();

		float rUnit = rBase - taper*(i+1);	// calculate radius of current unit	
		float mUnit = pow(rUnit,2)*PI*hUnit*rho;	// calculate mass of current unit
		float iUnitZ = 1.f/12.f*mUnit*(3*pow(rUnit,2)+pow(hUnit,2)); // calculate inertia of principle axis of current unit
		float iUnitX = 0.5*mUnit*pow(rUnit,2);	// calculate inertia of secondary axis of current unit

		// generate shape for unit
		//btCollisionShape* unitShape = new btSphereShape(btScalar(hUnit/2));
		btCollisionShape* unitShape = new btCylinderShapeX(btVector3(btScalar(hUnit/2),btScalar(rUnit),btScalar(rUnit)));
		btScalar  mass(mUnit);


		btVector3 localInertia(iUnitZ,iUnitX,iUnitX);
		unitShape->calculateLocalInertia(mass,localInertia);

		if(i==0){
			unitTransform.setOrigin(btVector3(
								btScalar(hUnit/2),
								btScalar(0),
								btScalar(0)));
			unitTransform.getBasis().setEulerZYX(0,0,0);

		}
		else{
			unitTransform.setOrigin(btVector3(
								btScalar((3*hUnit/2+springLength)*(i+1)),
								btScalar(0),
								btScalar(0)));
			unitTransform.getBasis().setEulerZYX(0,0,0);
		}
		


		whisker.push_back(createRigidBody(mass,basePointTransform*unitTransform,unitShape));		 
	} 
	 
	std::cout << "Whisker body created." << std::endl;
	

	// set up constraint for whisker base
	// ===========================================================

	btRigidBody* baseUnit = whisker[0];

	btVector3 pivotInBaseUnit(0,0,0);
	btVector3 axisInBaseUnit(0,1,0);

	// localFrame = whisker[0]->getWorldTransform().inverse();
	// baseUnit->setActivationState(DISABLE_DEACTIVATION);

	// btTransform frameBaseUnit;
	// frameBaseUnit.setIdentity();

	btTransform frameBaseUnit(btQuaternion::getIdentity(),btVector3(0, 0, 0));
	btGeneric6DofConstraint* baseConstraint = new btGeneric6DofConstraint(*baseUnit,frameBaseUnit,false);
	// btHingeConstraint* baseConstraint = new btHingeConstraint(*baseUnit,pivotInBaseUnit,axisInBaseUnit);
	// btHingeConstraint* baseConstraint = new btHingeConstraint(*whisker[0], localFrame, false);

	// float	targetVelocity = 0.2f;
	// float	maxMotorImpulse = 0.1f;
	// baseConstraint->enableAngularMotor(true,targetVelocity,maxMotorImpulse);

	// baseConstraint->setLimit( -PI * 0.25f, PI * 0.25f );
	
	// btConeTwistConstraint* baseConstraint = new btConeTwistConstraint(*whisker[0], localFrame);
	// baseConstraint->setLimit(btScalar (PI/4), btScalar (PI/4), btScalar (0));


	// btTransform frameInA(btQuaternion::getIdentity(),btVector3(rBase/4,0,0));						//par body's COM to cur body's COM offset
	// btTransform frameInB(btQuaternion::getIdentity(),btVector3(-hUnit/2,0,0));							//cur body's COM to cur body's PIV offset

	// btFixedConstraint* baseConstraint = new btFixedConstraint(*unitA, *unitB,frameInA,frameInB);

	// baseConstraint->setLinearLowerLimit(btVector3(0.0,0,0));
	// baseConstraint->setLinearUpperLimit(btVector3(0.0,0,0));
	// baseConstraint->setAngularLowerLimit(btVector3(0.0,0.0,0.0));
	// baseConstraint->setAngularUpperLimit(btVector3(0.0,0.0,0.0));

	

	baseConstraint->setLinearLowerLimit(btVector3(0., 0., 0.));
	baseConstraint->setLinearUpperLimit(btVector3(0., 0., 0.));
	// baseConstraint->setAngularLowerLimit(btVector3(0.,0.,0.));
	// baseConstraint->setAngularUpperLimit(btVector3(0.,0.,0.));


	// baseConstraint->getRotationalLimitMotor(3)->m_enableMotor = true;
	// baseConstraint->getRotationalLimitMotor(3)->m_targetVelocity = 1.f;
	// baseConstraint->getRotationalLimitMotor(3)->m_maxMotorForce = 0.01f;
	// baseConstraint->getRotationalLimitMotor(3)->m_maxLimitForce = 1000.00f;

	// baseConstraint->getRotationalLimitMotor(4)->m_enableMotor = true;
	// baseConstraint->getRotationalLimitMotor(4)->m_targetVelocity = 1.f;
	// baseConstraint->getRotationalLimitMotor(4)->m_maxMotorForce = 0.01f;
	// baseConstraint->getRotationalLimitMotor(4)->m_maxLimitForce = 1000.00f;

	// baseConstraint->getRotationalLimitMotor(5)->m_enableMotor = true;
	// baseConstraint->getRotationalLimitMotor(5)->m_targetVelocity = 1.f;
	// baseConstraint->getRotationalLimitMotor(5)->m_maxMotorForce = 0.01f;
	// baseConstraint->getRotationalLimitMotor(5)->m_maxLimitForce = 1000.00f;

	m_dynamicsWorld->addConstraint(baseConstraint);
	baseConstraint->setDbgDrawSize(btScalar(5.f));

	// set up constraints for whisker links (between units)
	// ==============================================================

	// define limits of rotation in links
	float thetaMax = PI/4;
	float phiMax = PI/4;
	float zetaMax = PI/10; 

	// define spring parameters in link
	float stiffness = 200.f; 	// stiffness of link springs
	float damping = 200.0; 		// damping of link springs

	for(int i=0;i<nUnit-1;++i) {
		
		btRigidBody* unitA = whisker[i];   // get unit i
		btRigidBody* unitB = whisker[i+1]; // get unit i+1
		 		
		// unitA->setActivationState(DISABLE_DEACTIVATION);
		// unitA->setActivationState(DISABLE_DEACTIVATION);

		btTransform frameInA = btTransform::getIdentity();					
		btTransform frameInB = btTransform::getIdentity();					

		frameInA.setOrigin(btVector3(0,0,0));	// place frames in COM
		frameInB.setOrigin(btVector3(0,0,0));

		// create link (between units) constraint
		btGeneric6DofSpring2Constraint* link = new btGeneric6DofSpring2Constraint(*unitA, *unitB, frameInA,frameInB);
		
		// set constraint limits
		link->setLinearUpperLimit(btVector3(hUnit, -0.0, -0.0)); // lock the units by hUnit apart from each other (x direction)
		link->setLinearLowerLimit(btVector3(hUnit, 0.0, 0.0));

		link->setAngularLowerLimit(btVector3(-zetaMax, -thetaMax, -phiMax)); // allow rotation around all axes
		link->setAngularUpperLimit(btVector3(zetaMax, thetaMax, phiMax));

		// add constraint to world
		m_dynamicsWorld->addConstraint(link, true); // ture -> collision between linked bodies disabled
		link->setDbgDrawSize(btScalar(5.f));
		
		// set spring parameters of links - are not physical measures
		// ----------------------------------------------------------

		// set parameters for rotation around x axis
		link->enableSpring(3, true);		
		link->setStiffness(3, stiffness);
		link->setDamping(3, damping);
		link->setEquilibriumPoint(3,0.);

		// set parameters for rotation around y axis (vertical)
		link->enableSpring(4, true);		
		link->setStiffness(4, stiffness);
		link->setDamping(4, damping);
		link->setEquilibriumPoint(4,0.);

		// set parameters for rotation around z axis
		link->enableSpring(5, true);		
		link->setStiffness(5, stiffness);
		link->setDamping(5, damping);
		link->setEquilibriumPoint(5,0.);
		
	}

	// m_once = true;

}

// void Whiskers::setWhiskerMotion(btScalar deltaTime, btHingeConstraint* baseLink)
// {

// 	float ms = deltaTime*1000000.;
// 	float minFPS = 1000000.f/60.f;
// 	if (ms > minFPS)
// 		ms = minFPS;

// 	m_Time += ms;

// 	//
// 	// set per-frame sinusoidal position targets using angular motor (hacky?)
// 	//	
// 	// for (int r=0; r<whiskerArray.size(); r++)
// 	// {
		
// 		btHingeConstraint* hingeC = baseLink;
// 		btScalar fCurAngle      = hingeC->getHingeAngle();
		
// 		btScalar fTargetPercent = (int(m_Time / 1000) % int(m_fCyclePeriod)) / m_fCyclePeriod;
// 		btScalar fTargetAngle   = 0.5 * (1 + sin(2 * M_PI * fTargetPercent));
// 		btScalar fTargetLimitAngle = hingeC->getLowerLimit() + fTargetAngle * (hingeC->getUpperLimit() - hingeC->getLowerLimit());
// 		btScalar fAngleError  = fTargetLimitAngle - fCurAngle;
// 		btScalar fDesiredAngularVel = 1000000.f * fAngleError/ms;
// 		hingeC->enableAngularMotor(true, fDesiredAngularVel, m_fMuscleStrength);
		
// 	// }

	
// }

void Whiskers::renderScene()
{
	CommonRigidBodyBase::renderScene();
	
}



CommonExampleInterface*    WhiskersCreateFunc(CommonExampleOptions& options)
{
	return new Whiskers(options.m_guiHelper);

}


B3_STANDALONE_EXAMPLE(WhiskersCreateFunc)



