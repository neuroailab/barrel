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
#ifndef WHISKER_HPP
#define WHISKER_HPP

#include "Whisker_utility.hpp"
#include "Whisker_config.hpp"
#include "Simulation_utility.hpp"
#include "Parameters.hpp"
#include <vector>
#include <string>


class Whisker
{
private:
	btDiscreteDynamicsWorld* m_dynamicsWorld;
	btAlignedObjectArray<btCollisionShape*>* m_collisionShapes;

	btRigidBody* refBody;
	btRigidBody* base;
	btAlignedObjectArray<btRigidBody*> whisker;

	btAlignedObjectArray<btGeneric6DofSpring2Constraint*> links;
	// btAlignedObjectArray<btJointFeedback*> links;

	btGeneric6DofSpring2Constraint* baseConstraint;
	btJointFeedback baseFeedback;

	float m_time;
	float m_angle;

	whisker_config config;
	Parameters* para;
	
public:

	Whisker(btDiscreteDynamicsWorld* world, btAlignedObjectArray<btCollisionShape*>* shapes, Parameters* parameters, btRigidBody* refBody);
	~Whisker(){}

	void createWhisker(std::string w_name);
	btRigidBody* get_unit(int idx);

	void moveWhisker(float time, float freq, float amp);
	
	std::vector<float> getBPTorques();
	std::vector<float> getBPForces();

	std::vector<float> getX();
	std::vector<float> getY();
	std::vector<float> getZ();
};

#endif //BASIC_DEMO_PHYSICS_SETUP_H
