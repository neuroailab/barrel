#ifndef WHISKER_UTILITY_HPP
#define WHISKER_UTILITY_HPP

#include "Simulation_utility.hpp"

#include "btBulletDynamicsCommon.h"
#include "btWorldImporter.h"

#include "LinearMath/btVector3.h"
#include "LinearMath/btAlignedObjectArray.h"
#include <iostream>

#include "CommonInterfaces/CommonRigidBodyBase.h"
#include "CommonInterfaces/CommonParameterInterface.h"

#include <vector>
#include <numeric>
#include <algorithm>
#include <map>
#include <string>
#include "math.h"



float calc_x_from_a_and_s(float a, float s);
std::vector<float> get_angles_from_curvature(float L, float a, int numLinks);

float calc_base_radius(int row, int col);
std::vector<float> calc_unit_radius(int numUnits, float rbase, float rtip);
std::vector<float> calc_mass(int numUnits, float unit_length, float rbase, float rtip, float rho);
std::vector<float> calc_inertia(int numUnits, float rbase, float rtip);
std::vector<float> calc_young_modulus(int numUnits, float E_base, float E_tip);
std::vector<float> calc_com(float L, float N, float rbase, float taper);
std::vector<float> calc_damping(int numLinks, std::vector<float> k, std::vector<float> mass, float CoM, float zeta_base, float zeta_tip);
std::vector<float> calc_stiffness(int numLinks, std::vector<float> E, std::vector<float> I, float unit_length);

btScalar getERP(btScalar timeStep, btScalar kSpring,btScalar kDamper);
btScalar getCFM(btScalar avoidSingularity, btScalar timeStep, btScalar kSpring,btScalar kDamper);


#endif //BASIC_DEMO_PHYSICS_SETUP_H
