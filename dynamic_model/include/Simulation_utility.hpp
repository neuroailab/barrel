#ifndef SIMULATION_UTILITY_HPP
#define SIMULATION_UTILITY_HPP


#define SCALE 100.
#define PI 3.1415927

#include <iostream>
#include <vector>

#include "btBulletDynamicsCommon.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "CommonInterfaces/CommonRigidBodyBase.h"
#include "CommonInterfaces/CommonParameterInterface.h"

#include "BulletDynamics/MLCPSolvers/btDantzigSolver.h"
#include "BulletDynamics/MLCPSolvers/btSolveProjectedGaussSeidel.h"
#include "BulletDynamics/MLCPSolvers/btMLCPSolver.h"
#include "BulletDynamics/ConstraintSolver/btNNCGConstraintSolver.h"


std::vector<float> btVecToFloat(btVector3 btVec);

btCollisionShape* createCylinderShape(btVector3 dimensions);
btCollisionShape* createSphereShape(btScalar radius);
btConvexHullShape* createFrostumShapeX(btScalar height, btScalar r_base, btScalar r_top, int NUM_POINTS);
btRigidBody* createDynamicBody(float mass, const btTransform& startTransform, btCollisionShape* shape,  const btVector4& color = btVector4(1, 0, 0, 1));

void translateFrame(btTransform& transform, btVector3 origin=btVector3(0.,0.,0.));
void rotateFrame(btTransform& transform, btVector3 rotation=btVector3(0.,0.,0.));
btTransform createFrame(btVector3 origin=btVector3(0.,0.,0.), btVector3 rotation=btVector3(0.,0.,0.));


#endif