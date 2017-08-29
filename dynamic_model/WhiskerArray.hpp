
#ifndef WHISKERARRAY_HPP
#define WHISKERARRAY_HPP


#include "Whisker.hpp"
#include <vector>
#include <string>

class WhiskerArray
{
private:

    btRigidBody* center;
	btAlignedObjectArray<Whisker*> m_whiskerArray;

	btDiscreteDynamicsWorld* m_dynamicsWorld;
	btAlignedObjectArray<btCollisionShape*>* m_collisionShapes;
	
	
	
public:

	WhiskerArray(btDiscreteDynamicsWorld* world, btAlignedObjectArray<btCollisionShape*>* shapes, Parameters* parameters);
	~WhiskerArray(){}

	void moveArray(float time, float freq, float angle_max);
    void moveCenter(float time, float freq, float angle_max);
    btAlignedObjectArray<Whisker*> getArray();
    void applyForceToWhisker(int idx_whisker, int idx_unit, btVector3 force);
    void releaseForceToWhisker(int idx_whisker, int idx_unit, btVector3 gravity);

	std::vector<std::vector<float>> getAllBPTorques();
	std::vector<std::vector<float>> getAllBPForces();
	std::vector<std::vector<float>> getAllX();
	std::vector<std::vector<float>> getAllY();
	std::vector<std::vector<float>> getAllZ();

	// std::vector<std::string> whisker_names={
	// "LA0",
 //    "LA1",
 //    "LA2",
 //    "LA3",
 //    "LA4",
 //    "LB0",
 //    "LB1",
 //    "LB2",
 //    "LB3",
 //    "LB4",
 //    "LB5",
 //    "LC0",
 //    "LC1",
 //    "LC2",
 //    "LC3",
 //    "LC4",
 //    "LC5",
 //    "LC6",
 //    "LD0",
 //    "LD1",
 //    "LD2",
 //    "LD3",
 //    "LD4",
 //    "LD5",
 //    "LD6",
 //    "LE1",
 //    "LE2",
 //    "LE3",
 //    "LE4",
 //    "LE5",
 //    "LE6",
 //    "RA0",
 //    "RA1",
 //    "RA2",
 //    "RA3",
 //    "RA4",
 //    "RB0",
 //    "RB1",
 //    "RB2",
 //    "RB3",
 //    "RB4",
 //    "RB5",
 //    "RC0",
 //    "RC1",
 //    "RC2",
 //    "RC3",
 //    "RC4",
 //    "RC5",
 //    "RC6",
 //    "RD0",
 //    "RD1",
 //    "RD2",
 //    "RD3",
 //    "RD4",
 //    "RD5",
 //    "RD6",
 //    "RE1",
 //    "RE2",
 //    "RE3",
 //    "RE4",
 //    "RE5",
 //    "RE6"
	// };

};




#endif //BASIC_DEMO_PHYSICS_SETUP_H