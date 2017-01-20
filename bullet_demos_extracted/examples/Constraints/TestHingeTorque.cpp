#include "TestHingeTorque.h"
#include <boost/program_options.hpp>

#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <cmath>

namespace po = boost::program_options;
using namespace std;

#include "../CommonInterfaces/CommonRigidBodyBase.h"
#include "../CommonInterfaces/CommonParameterInterface.h"

short collisionFilterGroup = short(btBroadphaseProxy::CharacterFilter);
short collisionFilterMask = short(btBroadphaseProxy::AllFilter ^ (btBroadphaseProxy::CharacterFilter));

vector<float> x_pos_base    = {-0.4};
vector<float> y_pos_base    = {7};
vector<float> z_pos_base    = {0};
vector<int> const_numLinks  = {15};
vector<float> qua_a_list    = {-1};
vector<float> yaw_y_base    = {0.5};
vector<float> pitch_x_base  = {0};
vector<float> roll_z_base   = {0};

const float SMALL_NUM = 1e-6;

float x_len_link    = 0.53;
float y_len_link    = 2.08;
float z_len_link    = 0.3;
float radius        = 1;
float basic_str     = 3000;
float linear_damp   = 0.77;
float ang_damp      = 0.79;
float time_leap     = 1.0/240.0;
//int limit_numLink   = const_numLinks + 1;
float equi_angle    = 0;

vector<int> inter_spring = {5};
vector<int> every_spring = {1};

float spring_stiffness = 520;
float spring_stfperunit     = 1000;
float camera_dist     = 45;
float spring_offset   = 0;
float time_limit    = 5.0/4;
float initial_str   = 10000;
float initial_stime = 1.0/8;
int initial_poi     = 14;
int flag_time       = 0;
float limit_softness = 0.9;
float limit_bias    = 0.3;
float limit_relax   = 1;
float limit_low     = -2;
float limit_up      = 0;
float state_ban_limit   = 1;
float velo_ban_limit    = 1;
float angl_ban_limit    = 1;
float force_limit       = 1;
float torque_limit      = 1;

int hinge_mode          = 1;
int test_mode           = 0;

float percision         = 0.001;

struct TestHingeTorque : public CommonRigidBodyBase{
    bool m_once;
    float pass_time;
    float curr_velo;
    float curr_angl;
    float curr_state;
    float curr_force;
    float curr_torque;

    btAlignedObjectArray< btAlignedObjectArray< btRigidBody* > > m_allbones_big_list;
    btAlignedObjectArray< btAlignedObjectArray< btHingeConstraint* > > m_allhinges_big_list;
    btAlignedObjectArray< btAlignedObjectArray<btJointFeedback*> > m_jointFeedback_big_list;

	TestHingeTorque(struct GUIHelperInterface* helper);
	virtual ~ TestHingeTorque();
	virtual void initPhysics();

	virtual void stepSimulation(float deltaTime);
    void addQuaUnits(float qua_a, float qua_b, float qua_c, int num_units, 
            btTransform base_transform);
    btJointFeedback* addFeedbackForSpring(btGeneric6DofSpring2Constraint* con);

	
	virtual void resetCamera(){
        //float dist = 5;
        float dist = camera_dist;
        //float dist = 15;
        float pitch = 270;
        float yaw = 21;
        //float targetPos[3]={-1.34,3.4,-0.44};
        float targetPos[3]={-1.34,5.4,-0.44};
        m_guiHelper->resetCamera(dist,pitch,yaw,targetPos[0],targetPos[1],targetPos[2]);
	}
};

TestHingeTorque::TestHingeTorque(struct GUIHelperInterface* helper)
:CommonRigidBodyBase(helper),
m_once(true){
}
TestHingeTorque::~ TestHingeTorque(){
    /*
	for (int i=0;i<m_jointFeedback.size();i++){
		delete m_jointFeedback[i];
	}
    */
}

void TestHingeTorque::stepSimulation(float deltaTime){
    
    m_dynamicsWorld->stepSimulation(time_leap,0);
    pass_time   = pass_time + time_leap;
    if (flag_time==1){
        if (pass_time > time_limit){
            exit(0);
        }
    }

    curr_velo   = 0; //Linear speed
    curr_angl   = 0; //angle speed
    curr_state  = 0; //Not used. Hinge's difference from their balance point
    curr_force  = 0; //force applied by springs 
    curr_torque = 0; // torque applied by springs

    static int count = 0;
    int all_size_for_big_list   = 0;
    int all_size_for_fb         = 0;

    for (int big_list_indx=0;big_list_indx < const_numLinks.size(); big_list_indx++) {
        btAlignedObjectArray< btRigidBody* > m_allbones;
        btAlignedObjectArray< btHingeConstraint* > m_allhinges;
        btAlignedObjectArray< btJointFeedback* > m_jointFeedback;

        m_allhinges.clear();

        m_allbones      = m_allbones_big_list[big_list_indx];
        m_jointFeedback = m_jointFeedback_big_list[big_list_indx];

        int all_size    = m_allbones.size();

        if (hinge_mode==1){
            m_allhinges     = m_allhinges_big_list[big_list_indx];
            all_size    = m_allhinges.size();
            for (int i=0;i<all_size;i++){
                btVector3 tmp_center_pos = m_allbones[i]->getCenterOfMassPosition();

                btTransform tmp_trans = m_allbones[i]->getCenterOfMassTransform();
                btVector3 tmp_pos = tmp_trans(btVector3(0,0,-z_len_link));
                btVector3 tmp_direction = tmp_center_pos - tmp_pos;
                //btVector3 tmp_pos_f = tmp_trans(btVector3(0,y_len_link,0));
                btVector3 tmp_pos_f = tmp_trans(btVector3(0,-y_len_link,0));
                btVector3 tmp_point     = tmp_pos_f - tmp_center_pos;

                btScalar tmp_angle_high     = m_allhinges[i]->getHingeAngle() - equi_angle;

                btVector3 tmp_force_now     = tmp_direction*tmp_angle_high*basic_str*(all_size - i)/2;
                m_allbones[i]->applyForce(tmp_force_now, -tmp_point);
                //m_allbones[i]->applyForce(tmp_force_now, tmp_point);

                //if (i < all_size -1){
                if (i < all_size){
                    btTransform tmp_trans = m_allbones[i+1]->getCenterOfMassTransform();
                    btVector3 tmp_pos = tmp_trans(btVector3(0,0,z_len_link));
                    //btVector3 tmp_pos_f = tmp_trans(btVector3(0,-y_len_link,0));
                    btVector3 tmp_pos_f = tmp_trans(btVector3(0,y_len_link,0));
                    btVector3 tmp_point     = tmp_pos_f - tmp_center_pos;
                    m_allbones[i+1]->applyForce(-tmp_force_now, -tmp_point);
                    //m_allbones[i+1]->applyForce(-tmp_force_now, tmp_point);
                }

            }
        }

        for (int i=0;i<all_size;i++){
            curr_velo   += m_allbones[i]->getLinearVelocity().norm();
            curr_angl   += m_allbones[i]->getAngularVelocity().norm();
            //curr_force  += m_allbones[i]->getTotalForce().norm();
            //curr_torque += m_allbones[i]->getTotalTorque().norm();
        }

        for (int i=0;i<m_jointFeedback.size();i++){
            curr_force  += m_jointFeedback[i]->m_appliedForceBodyA.norm();
            curr_force  += m_jointFeedback[i]->m_appliedForceBodyB.norm();

            curr_torque += m_jointFeedback[i]->m_appliedTorqueBodyA.norm();
            curr_torque += m_jointFeedback[i]->m_appliedTorqueBodyB.norm();
        }

        if ((pass_time < initial_stime) && (initial_poi < all_size-1)){
            m_allbones[initial_poi+1]->applyForce(btVector3(0,0,initial_str), btVector3(0,0,0));
            cout << "Applied" << endl;
        }

        for (int i=0;i<m_allhinges.size();i++){
            curr_state  += m_allhinges[i]->getHingeAngle() - limit_up;
        }

        all_size_for_big_list   += all_size;
        all_size_for_fb         += m_jointFeedback.size();
    }
    count++;
    curr_velo   /= all_size_for_big_list;
    curr_angl   /= all_size_for_big_list;
    curr_state  /= all_size_for_big_list;
    curr_force  /= all_size_for_fb;
    curr_torque /= all_size_for_fb;

    if ((flag_time==2) && (pass_time > initial_stime)){
        if ((curr_velo < velo_ban_limit) && (curr_state < state_ban_limit) && (curr_angl < angl_ban_limit) && (curr_force < force_limit) && (curr_torque < torque_limit)){
            cout << "Now state:" << curr_velo << " " << curr_angl << " " << curr_force << " " << curr_torque << " " << curr_state << endl;
            exit(0);
        }
    }

}


float getValueQua(float qua_a, float qua_b, float qua_c, float x_now){
    float y_now = qua_a*pow(x_now, 2) + qua_b*x_now + qua_c;
    return y_now;
}

float getNorm(float x_now, float y_now){
    float norm_now = sqrt(pow(x_now, 2) + pow(y_now, 2));
    return norm_now;
}

float getDistanceQua(float qua_a, float qua_b, float qua_c, float cir_x0, float x_now){
    float cir_y0 = getValueQua(qua_a, qua_b, qua_c, cir_x0);
    float y_now = getValueQua(qua_a, qua_b, qua_c, x_now);
    float distance_now = getNorm(cir_x0 - x_now, cir_y0 - y_now);

    return distance_now;
}

float findInterQuaCirleBin(float qua_a, float qua_b, float qua_c, float cir_x0, float cir_r0, 
        float percision, float direction = 1){
    float curr_distance = 20*direction;
    while (getDistanceQua(qua_a, qua_b, qua_c, cir_x0, cir_x0 + curr_distance) < cir_r0) {
        curr_distance = curr_distance*2;
    }

    float max_dis = curr_distance;
    float min_dis = 0;

    float final_val = 0;

    while (true){
        float now_dis_s = getDistanceQua(qua_a, qua_b, qua_c, cir_x0, cir_x0 + min_dis);
        if (abs(now_dis_s - cir_r0) < percision){
            final_val = cir_x0 + min_dis;
            break;
        }
        float now_dis_b = getDistanceQua(qua_a, qua_b, qua_c, cir_x0, cir_x0 + max_dis);
        if (abs(now_dis_b - cir_r0) < percision){
            final_val = cir_x0 + max_dis;
            break;
        }

        curr_distance = (max_dis + min_dis)/2;
        if (getDistanceQua(qua_a, qua_b, qua_c, cir_x0, cir_x0 + curr_distance) < cir_r0) {
            min_dis = curr_distance;
        } else {
            max_dis = curr_distance;
        }
    }

    return final_val;
}

btVector3 findLineInter(btVector3 pre_1, btVector3 next_1, btVector3 pre_2, btVector3 next_2){
    float x_1 = pre_1[2], y_1 = pre_1[1], x_2 = next_1[2], y_2 = next_1[1];
    float x_p_1 = pre_2[2], y_p_1 = pre_2[1], x_p_2 = next_2[2], y_p_2 = next_2[1]; 

    float a_11 = x_2 - x_1, a_12 = y_1 - y_2;
    float a_21 = x_p_2 - x_p_1, a_22 = y_p_1 - y_p_2;

    //cout << a_11 << " " << a_12 << " " << a_21 << " " << a_22 << endl;

    float abs_mat = a_11*a_22 - a_21*a_12;

    if (abs(abs_mat) < SMALL_NUM){
        return (pre_2 + next_1)/2;
    }

    float b_1 = x_2*y_1 - x_1*y_2, b_2 = x_p_2*y_p_1 - x_p_1*y_p_2;

    //cout << b_1 << " " << b_2 << endl;

    float inv_abs = (1/abs_mat);

    float new_y = inv_abs*(a_22*b_1 - a_12*b_2), new_z = inv_abs*(a_11*b_2 - a_21*b_1);
    return btVector3(0, new_y, new_z);
}

btJointFeedback* TestHingeTorque::addFeedbackForSpring(btGeneric6DofSpring2Constraint* con){
    btJointFeedback* fb = new btJointFeedback();
    con->setJointFeedback(fb);
    return fb;
}

void TestHingeTorque::addQuaUnits(float qua_a, float qua_b, float qua_c, int num_units, 
        btTransform base_transform = btTransform(btQuaternion::getIdentity(),btVector3(0, 0, 0))){
    float pre_z     = 0;
    float pre_y     = getValueQua(qua_a, qua_b, qua_c, pre_z);
    float pre_deg   = 0;

    btRigidBody* pre_unit = 0;
        
    float baseMass = 0.f;
    float linkMass = 1.f;

    btAlignedObjectArray< btRigidBody* > m_allbones;
    btAlignedObjectArray< btVector3 > m_allpre;
    btAlignedObjectArray< btVector3 > m_allnext;
    btAlignedObjectArray<btJointFeedback*> m_jointFeedback;
    vector<float> m_alldeg;

    m_allbones.clear();
    m_allpre.clear();
    m_allnext.clear();
    m_alldeg.clear();

    btVector3 linkHalfExtents(x_len_link, y_len_link, z_len_link);
    btBoxShape* baseBox = new btBoxShape(linkHalfExtents);
    btSphereShape* linkSphere = new btSphereShape(radius);

    for (int indx_unit=0;indx_unit < num_units;indx_unit++){
        float next_z = findInterQuaCirleBin(qua_a, qua_b, qua_c, pre_z, 2*y_len_link, percision, -1);
        float next_y = getValueQua(qua_a, qua_b, qua_c, next_z);
        float base_y_tmp = (pre_y + next_y)/2;
        float base_z_tmp = (pre_z + next_z)/2;

        float deg_away = atan2(next_z - pre_z, next_y - pre_y);
        btQuaternion test_rotation = btQuaternion( 0, deg_away, 0);
        btVector3 basePosition = btVector3( 0, base_y_tmp, base_z_tmp);
        btTransform baseWorldTrans(test_rotation, basePosition);
        baseWorldTrans  = base_transform*baseWorldTrans;

        btRigidBody* base = createRigidBody(linkMass,baseWorldTrans,baseBox);

        m_dynamicsWorld->removeRigidBody(base);
        base->setDamping(linear_damp,ang_damp);
        m_dynamicsWorld->addRigidBody(base,collisionFilterGroup,collisionFilterMask);

        if (pre_unit){
            btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0, y_len_link, 0));						//par body's COM to cur body's COM offset
            btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0, -y_len_link, 0));							//cur body's COM to cur body's PIV offset
            btGeneric6DofSpring2Constraint* fixed = new btGeneric6DofSpring2Constraint(*pre_unit, *base, pivotInA, pivotInB);

            for (int indx_tmp=3;indx_tmp<6;indx_tmp++){
                fixed->enableSpring(indx_tmp, true);
                fixed->setStiffness(indx_tmp, basic_str + spring_stfperunit*(num_units - indx_unit -1));
            }

            fixed->setEquilibriumPoint(3, -(deg_away - pre_deg));

            for (int indx_tmp=0;indx_tmp<3;indx_tmp++){
                fixed->enableSpring(indx_tmp, true);
                fixed->setStiffness(indx_tmp, basic_str + spring_stfperunit*(num_units - indx_unit -1));
            }

            m_jointFeedback.push_back(addFeedbackForSpring(fixed));

            m_dynamicsWorld->addConstraint(fixed,true);

            btVector3 axisInA(1,0,0);
            btVector3 axisInB(1,0,0);
            btVector3 pivotInA_h(0, y_len_link,0);
            btVector3 pivotInB_h(0,-y_len_link,0);
            bool useReferenceA = true;
            btHingeConstraint* hinge = new btHingeConstraint(*pre_unit,*base,
                pivotInA_h,pivotInB_h,
                axisInA,axisInB,useReferenceA);
            m_dynamicsWorld->addConstraint(hinge,true);
        } else {
            btVector3 basePosition_ball = btVector3(0, -radius-y_len_link, 0);
            btTransform baseWorldTrans_ball;
            baseWorldTrans_ball.setIdentity();
            baseWorldTrans_ball.setOrigin(basePosition_ball);
            baseWorldTrans_ball     = baseWorldTrans*baseWorldTrans_ball;

            btRigidBody* base_ball = createRigidBody(baseMass,baseWorldTrans_ball,linkSphere);
            m_dynamicsWorld->removeRigidBody(base_ball);
            base_ball->setDamping(0,0);
            m_dynamicsWorld->addRigidBody(base_ball,collisionFilterGroup,collisionFilterMask);

            // Special spring for base ball and base box unit 
            btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0, radius, 0));						//par body's COM to cur body's COM offset
            btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0, -y_len_link, 0));							//cur body's COM to cur body's PIV offset
            btGeneric6DofSpring2Constraint* fixed = new btGeneric6DofSpring2Constraint(*base_ball, *base,pivotInA,pivotInB);
            fixed->setLinearLowerLimit(btVector3(0,0,0));
            fixed->setLinearUpperLimit(btVector3(0,0,0));
            fixed->setAngularLowerLimit(btVector3(0,-1,0));
            fixed->setAngularUpperLimit(btVector3(0,1,0));
            for (int indx_tmp=3;indx_tmp<6;indx_tmp++){
                fixed->enableSpring(indx_tmp, true);
                fixed->setStiffness(indx_tmp, basic_str + spring_stfperunit*(num_units - indx_unit -1));
            }
            fixed->setEquilibriumPoint();

            m_jointFeedback.push_back(addFeedbackForSpring(fixed));

            m_dynamicsWorld->addConstraint(fixed,true);
        }


        btVector3 curr_pre  = btVector3(0, pre_y, pre_z);
        btVector3 curr_next = btVector3(0, next_y, next_z);

        // Create more spings between distant units
        for (int spring_indx=0;spring_indx < every_spring.size();spring_indx++){
            int tmp_every_spring    = every_spring[spring_indx];
            int tmp_inter_spring    = inter_spring[spring_indx];

            if ((indx_unit>=tmp_every_spring) && (indx_unit % tmp_inter_spring==0)){

                int new_indx = indx_unit - tmp_every_spring;

                btVector3 inter_point = findLineInter(m_allpre[new_indx], m_allnext[new_indx], curr_pre, curr_next);

                btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0,  inter_point.distance((m_allpre[new_indx] + m_allnext[new_indx])/2), 0));
                btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0, -inter_point.distance((curr_pre + curr_next)/2), 0));

                btGeneric6DofSpring2Constraint* fixed;

                fixed = new btGeneric6DofSpring2Constraint(*m_allbones[new_indx], *base, pivotInA,pivotInB);

                for (int indx_tmp=0;indx_tmp<6;indx_tmp++){
                    fixed->enableSpring(indx_tmp, true);
                    fixed->setStiffness(indx_tmp, spring_stiffness + (num_units - (new_indx + indx_unit)/2)*spring_stfperunit);
                }
                fixed->setEquilibriumPoint(3, -(deg_away - m_alldeg[new_indx]));
                

                m_jointFeedback.push_back(addFeedbackForSpring(fixed));
                m_dynamicsWorld->addConstraint(fixed,true);
            }
        }

        m_allbones.push_back(base);
        m_allpre.push_back(curr_pre);
        m_allnext.push_back(curr_next);
        m_alldeg.push_back(deg_away);

        pre_unit    = base;
        pre_y       = next_y;
        pre_z       = next_z;
        pre_deg     = deg_away;
    }

    m_allbones_big_list.push_back(m_allbones);
    m_jointFeedback_big_list.push_back(m_jointFeedback);
}


void TestHingeTorque::initPhysics(){
	int upAxis = 1;
    pass_time   = 0;
    m_allbones_big_list.clear();
    m_allhinges_big_list.clear();
    
    b3Printf("Config name = %s",m_guiHelper->getconfigname());
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("x_pos_base", po::value<vector<float>>()->multitoken(), "Position x of base")
            ("y_pos_base", po::value<vector<float>>()->multitoken(), "Position y of base")
            ("z_pos_base", po::value<vector<float>>()->multitoken(), "Position z of base")
            ("const_numLinks", po::value<vector<int>>()->multitoken(), "Number of units")
            ("qua_a_list", po::value<vector<float>>()->multitoken(), "Quadratic Coefficient")
            ("yaw_y_base", po::value<vector<float>>()->multitoken(), "Parameter yaw for btQuaternion")
            ("pitch_x_base", po::value<vector<float>>()->multitoken(), "Parameter pitch for btQuaternion")
            ("roll_z_base", po::value<vector<float>>()->multitoken(), "Parameter roll for btQuaternion")
            ("x_len_link", po::value<float>(), "Size x of cubes")
            ("y_len_link", po::value<float>(), "Size y of cubes")
            ("z_len_link", po::value<float>(), "Size z of cubes")
            ("radius", po::value<float>(), "Size of the ball at top")
            ("basic_str", po::value<float>(), "Minimal strength of hinge's recover force")
            ("linear_damp", po::value<float>(), "Control the linear damp ratio")
            ("ang_damp", po::value<float>(), "Control the angle damp ratio")
            ("time_leap", po::value<float>(), "Time unit for simulation")
            ("equi_angle", po::value<float>(), "Control the angle of balance for hinges")
            ("inter_spring", po::value<vector<int>>()->multitoken(), "Number of units between two strings")
            ("every_spring", po::value<vector<int>>()->multitoken(), "Number of units between one strings")
            ("spring_stiffness", po::value<float>(), "Stiffness of spring")
            ("spring_stfperunit", po::value<float>(), "Stiffness of spring")
            ("camera_dist", po::value<float>(), "Distance of camera")
            ("spring_offset", po::value<float>(), "String offset for balance state")
            ("time_limit", po::value<float>(), "Time limit for recording")
            ("initial_str", po::value<float>(), "Initial strength of force applied")
            ("initial_stime", po::value<float>(), "Initial time to apply force")
            ("initial_poi", po::value<int>(), "Unit to apply the force")
            ("flag_time", po::value<int>(), "Whether open time limit")
            ("limit_softness", po::value<float>(), "Softness of the hinge limit")
            ("limit_bias", po::value<float>(), "Bias of the hinge limit")
            ("limit_relax", po::value<float>(), "Relax of the hinge limit")
            ("limit_low", po::value<float>(), "Lower bound of the hinge limit")
            ("limit_up", po::value<float>(), "Up bound of the hinge limit")
            ("state_ban_limit", po::value<float>(), "While flag_time is 2, used for angle states of hinges to judge whether stop")
            ("velo_ban_limit", po::value<float>(), "While flag_time is 2, used for linear velocities of rigid bodys to judge whether stop")
            ("angl_ban_limit", po::value<float>(), "While flag_time is 2, used for angular velocities of rigid bodys to judge whether stop")
            ("force_limit", po::value<float>(), "While flag_time is 2, used for forces of rigid bodys to judge whether stop")
            ("torque_limit", po::value<float>(), "While flag_time is 2, used for torques of rigid bodys to judge whether stop")
            ("hinge_mode", po::value<int>(), "Whether use hinges rather than springs for connections of two units")
            ("test_mode", po::value<int>(), "Whether enter test mode for some temp test codes, default is 0")
        ;

        po::variables_map vm;
        const char* file_name = m_guiHelper->getconfigname();
        ifstream ifs(file_name);
        po::store(po::parse_config_file(ifs, desc), vm);
        po::notify(vm);    

        if (vm.count("x_pos_base")){
            x_pos_base      = vm["x_pos_base"].as< vector<float> >();
        }
        if (vm.count("y_pos_base")){
            y_pos_base      = vm["y_pos_base"].as< vector<float> >();
        }
        if (vm.count("z_pos_base")){
            z_pos_base      = vm["z_pos_base"].as< vector<float> >();
        }
        if (vm.count("const_numLinks")){
            const_numLinks  = vm["const_numLinks"].as< vector<int> >();
        }
        if (vm.count("qua_a_list")){
            qua_a_list      = vm["qua_a_list"].as< vector<float> >();
        }
        if (vm.count("yaw_y_base")){
            yaw_y_base      = vm["yaw_y_base"].as< vector<float> >();
        }
        if (vm.count("pitch_x_base")){
            pitch_x_base    = vm["pitch_x_base"].as< vector<float> >();
        }
        if (vm.count("roll_z_base")){
            roll_z_base     = vm["roll_z_base"].as< vector<float> >();
        }

        //limit_numLink = const_numLinks + 1;
        if ((x_pos_base.size()!=y_pos_base.size()) || (y_pos_base.size()!=z_pos_base.size()) || (x_pos_base.size()!=const_numLinks.size()) || (x_pos_base.size()!=qua_a_list.size())){
            cerr << "error: size not equal for (xyz,num)!" << endl;
            exit(0);
        }

        if (vm.count("x_len_link")){
            x_len_link      = vm["x_len_link"].as<float>();
        }
        if (vm.count("y_len_link")){
            y_len_link      = vm["y_len_link"].as<float>();
        }
        if (vm.count("z_len_link")){
            z_len_link      = vm["z_len_link"].as<float>();
        }
        if (vm.count("radius")){
            radius          = vm["radius"].as<float>();
        }

        if (vm.count("basic_str")){
            basic_str       = vm["basic_str"].as<float>();
        }


        if (vm.count("linear_damp")){
            linear_damp     = vm["linear_damp"].as<float>();
        }
        if (vm.count("ang_damp")){
            ang_damp        = vm["ang_damp"].as<float>();
        }

        if (vm.count("time_leap")){
            time_leap       = vm["time_leap"].as<float>();
        }

        if (vm.count("equi_angle")){
            equi_angle      = vm["equi_angle"].as<float>();
        }

        if (vm.count("inter_spring")){
            inter_spring    = vm["inter_spring"].as< vector<int> >();
        }
        if (vm.count("every_spring")){
            every_spring    = vm["every_spring"].as< vector<int> >();
        }

        if (every_spring.size()!=inter_spring.size()){
            cerr << "error: size not equal!" << endl;
            exit(0);
        }

        if (vm.count("spring_stiffness")){
            spring_stiffness    = vm["spring_stiffness"].as<float>();
        }
        if (vm.count("spring_stfperunit")){
            spring_stfperunit   = vm["spring_stfperunit"].as<float>();
        }
        if (vm.count("camera_dist")){
            camera_dist     = vm["camera_dist"].as<float>();
        }
        if (vm.count("spring_offset")){
            spring_offset   = vm["spring_offset"].as<float>();
        }
        if (vm.count("time_limit")){
            time_limit      = vm["time_limit"].as<float>();
        }
        if (vm.count("initial_str")){
            initial_str     = vm["initial_str"].as<float>();
        }
        if (vm.count("initial_stime")){
            initial_stime   = vm["initial_stime"].as<float>();
        }
        if (vm.count("initial_poi")){
            initial_poi     = vm["initial_poi"].as<int>();
        }
        if (vm.count("flag_time")){
            flag_time       = vm["flag_time"].as<int>();
        }

        if (vm.count("limit_softness")){
            limit_softness  = vm["limit_softness"].as<float>();
        }
        if (vm.count("limit_bias")){
            limit_bias      = vm["limit_bias"].as<float>();
        }
        if (vm.count("limit_relax")){
            limit_relax     = vm["limit_relax"].as<float>();
        }
        if (vm.count("limit_low")){
            limit_low       = vm["limit_low"].as<float>();
        }
        if (vm.count("limit_up")){
            limit_up        = vm["limit_up"].as<float>();
        }

        if (vm.count("state_ban_limit")){
            state_ban_limit = vm["state_ban_limit"].as<float>();
        }
        if (vm.count("velo_ban_limit")){
            velo_ban_limit  = vm["velo_ban_limit"].as<float>();
        }
        if (vm.count("angl_ban_limit")){
            angl_ban_limit  = vm["angl_ban_limit"].as<float>();
        }
        if (vm.count("force_limit")){
            force_limit     = vm["force_limit"].as<float>();
        }
        if (vm.count("torque_limit")){
            torque_limit    = vm["torque_limit"].as<float>();
        }

        if (vm.count("hinge_mode")){
            hinge_mode      = vm["hinge_mode"].as<int>();
        }

        if (vm.count("test_mode")){
            test_mode       = vm["test_mode"].as<int>();
        }
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }
    
	m_guiHelper->setUpAxis(upAxis);

	createEmptyDynamicsWorld();
	m_dynamicsWorld->getSolverInfo().m_splitImpulse = false;
	
    m_dynamicsWorld->setGravity(btVector3(0,0,0));
    
	m_guiHelper->createPhysicsDebugDrawer(m_dynamicsWorld);
	int mode = 	btIDebugDraw::DBG_DrawWireframe
				+btIDebugDraw::DBG_DrawConstraints
				+btIDebugDraw::DBG_DrawConstraintLimits;
	m_dynamicsWorld->getDebugDrawer()->setDebugMode(mode);


    btVector3 linkHalfExtents(x_len_link, y_len_link, z_len_link);

    btBoxShape* baseBox = new btBoxShape(linkHalfExtents);
    btBoxShape* linkBox1 = new btBoxShape(linkHalfExtents);
    btSphereShape* linkSphere = new btSphereShape(radius);

    if (test_mode==1) {

        //btVector3 test_v = findLineInter(btVector3(0,0,0), btVector3(0,1,0), btVector3(0,3,-1), btVector3(0,4,-2));
        //cout << test_v[1] << " " << test_v[2] << endl;
        //addQuaUnits(-1, 0, 0, 4, btTransform(btQuaternion::getIdentity(),btVector3(x_pos_base[0], y_pos_base[0], z_pos_base[0])));
        addQuaUnits(qua_a_list[0], 0, 0, const_numLinks[0], btTransform(btQuaternion( yaw_y_base[0], pitch_x_base[0], roll_z_base[0]),btVector3(x_pos_base[0], y_pos_base[0], z_pos_base[0])));
        //addQuaUnits(-1, 0, y_pos_base[0], 4, btTransform(btQuaternion::getIdentity(),btVector3(0, 7, 0)));

    } else {
        for (int big_list_indx=0;big_list_indx < const_numLinks.size(); big_list_indx++) { // create one single whisker 
            addQuaUnits(qua_a_list[big_list_indx], 0, 0, const_numLinks[big_list_indx], btTransform(btQuaternion( yaw_y_base[big_list_indx], pitch_x_base[big_list_indx], roll_z_base[big_list_indx]),btVector3(x_pos_base[big_list_indx], y_pos_base[big_list_indx], z_pos_base[big_list_indx])));
        }
        /*
        for (int big_list_indx=0;big_list_indx < const_numLinks.size(); big_list_indx++) { // create one single whisker 

            btAlignedObjectArray< btRigidBody* > m_allbones;
            btAlignedObjectArray< btHingeConstraint* > m_allhinges;

            m_allbones.clear();
            m_allhinges.clear();
            
            int numLinks = const_numLinks[big_list_indx];
            btVector3 basePosition = btVector3( x_pos_base[big_list_indx], y_pos_base[big_list_indx], z_pos_base[big_list_indx]);
            btTransform baseWorldTrans;
            baseWorldTrans.setIdentity();
            baseWorldTrans.setOrigin(basePosition);
            
            float baseMass = 0.f;
            float linkMass = 1.f;
            
            // Create the base ball
            btVector3 basePosition_ball = btVector3(x_pos_base[big_list_indx], y_pos_base[big_list_indx] + y_len_link + radius, z_pos_base[big_list_indx]);
            btTransform baseWorldTrans_ball;
            baseWorldTrans_ball.setIdentity();
            baseWorldTrans_ball.setOrigin(basePosition_ball);

            btRigidBody* base_ball = createRigidBody(baseMass,baseWorldTrans_ball,linkSphere);
            m_dynamicsWorld->removeRigidBody(base_ball);
            base_ball->setDamping(0,0);
            m_dynamicsWorld->addRigidBody(base_ball,collisionFilterGroup,collisionFilterMask);
            
            // Create the base box
            btRigidBody* base = createRigidBody(linkMass,baseWorldTrans,baseBox);
            m_dynamicsWorld->removeRigidBody(base);
            base->setDamping(linear_damp,ang_damp);
            m_dynamicsWorld->addRigidBody(base,collisionFilterGroup,collisionFilterMask);

            // Special spring for base ball and base box unit 
            btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0, -radius, 0));						//par body's COM to cur body's COM offset
            btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0, radius, 0));							//cur body's COM to cur body's PIV offset
            btGeneric6DofSpring2Constraint* fixed = new btGeneric6DofSpring2Constraint(*base_ball, *base,pivotInA,pivotInB);
            fixed->setLinearLowerLimit(btVector3(0,0,0));
            fixed->setLinearUpperLimit(btVector3(0,0,0));
            fixed->setAngularLowerLimit(btVector3(0,-1,0));
            fixed->setAngularUpperLimit(btVector3(0,1,0));
            for (int indx_tmp=3;indx_tmp<6;indx_tmp++){
                fixed->enableSpring(indx_tmp, true);
                fixed->setStiffness(indx_tmp, basic_str*numLinks/2);
            }
            fixed->setEquilibriumPoint();

            m_dynamicsWorld->addConstraint(fixed,true);

            btRigidBody* prevBody = base;
            m_allbones.push_back(base);

            cout << "Push base" << endl;
            
            for (int i=0;i<numLinks;i++){
                btTransform linkTrans;
                linkTrans = baseWorldTrans;
                
                linkTrans.setOrigin(basePosition-btVector3(0,linkHalfExtents[1]*2.f*(i+1),0));
                
                btCollisionShape* colOb = 0;
                
                colOb = linkBox1;

                btRigidBody* linkBody = createRigidBody(linkMass,linkTrans,colOb);
                m_dynamicsWorld->removeRigidBody(linkBody);
                linkBody->setDamping(linear_damp,ang_damp);
                m_dynamicsWorld->addRigidBody(linkBody,collisionFilterGroup,collisionFilterMask);

                m_allbones.push_back(linkBody);

                btTypedConstraint* con = 0;
                
                if (1) {
                    //create a hinge constraint
                    if (hinge_mode==1){
                        btVector3 pivotInA(0,-linkHalfExtents[1],0);
                        btVector3 pivotInB(0,linkHalfExtents[1],0);
                        btVector3 axisInA(1,0,0);
                        btVector3 axisInB(1,0,0);
                        bool useReferenceA = true;
                        btHingeConstraint* hinge = new btHingeConstraint(*prevBody,*linkBody,
                            pivotInA,pivotInB,
                            axisInA,axisInB,useReferenceA);
                        hinge->setLimit(limit_low, limit_up, limit_softness, limit_bias, limit_relax);
                        m_allhinges.push_back(hinge);
                        con = hinge;
                    } else {
                        btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0, -linkHalfExtents[1], 0));
                        btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0,  linkHalfExtents[1], 0));
                        btGeneric6DofSpring2Constraint* fixed = new btGeneric6DofSpring2Constraint(*prevBody, *linkBody,pivotInA,pivotInB);
                        fixed->setLinearLowerLimit(btVector3(0,0,0));
                        fixed->setLinearUpperLimit(btVector3(0,0,0));
                        fixed->setAngularLowerLimit(btVector3( limit_low,0,0));
                        fixed->setAngularUpperLimit(btVector3(  limit_up,0,0));
                        for (int indx_tmp=3;indx_tmp<6;indx_tmp++){
                            fixed->enableSpring(indx_tmp, true);
                            fixed->setStiffness(indx_tmp, basic_str*(numLinks-i)/2);
                        }
                        for (int indx_tmp=0;indx_tmp<6;indx_tmp++){
                            if (indx_tmp!=3) {
                                fixed->setEquilibriumPoint(indx_tmp);
                            } else {
                                if (limit_low > 0)
                                    fixed->setEquilibriumPoint(indx_tmp, limit_low);
                                else if (limit_up < 0)
                                    fixed->setEquilibriumPoint(indx_tmp, limit_up);
                                else
                                    fixed->setEquilibriumPoint(indx_tmp);
                            }
                        }
                        //fixed->setEquilibriumPoint();
                        con = fixed;
                    }
                    
                    // Create a spring constraint if needed
                    for (int spring_indx=0;spring_indx < every_spring.size();spring_indx++){
                        int tmp_every_spring    = every_spring[spring_indx];
                        int tmp_inter_spring    = inter_spring[spring_indx];

                        if ((i>tmp_every_spring-2) && (i % tmp_inter_spring==0)){
                            btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0, -tmp_every_spring*linkHalfExtents[1]+spring_offset, 0));
                            btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0, tmp_every_spring*linkHalfExtents[1]-spring_offset, 0));
                            //btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0, 0, 0));
                            //btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0, 0, 0));
                            btGeneric6DofSpring2Constraint* fixed;

                            fixed = new btGeneric6DofSpring2Constraint(*m_allbones[i-(tmp_every_spring)+1], *linkBody,pivotInA,pivotInB);
                            //for (int indx_tmp=0;indx_tmp<3;indx_tmp++){
                            for (int indx_tmp=0;indx_tmp<6;indx_tmp++){
                                fixed->enableSpring(indx_tmp, true);
                                fixed->setStiffness(indx_tmp, spring_stiffness);
                            }
                            fixed->setEquilibriumPoint();
                            m_dynamicsWorld->addConstraint(fixed,true);
                        }
                    }
                }

                btAssert(con);
                if (con){
                    // btJointFeedback* fb = new btJointFeedback();
                    // m_jointFeedback.push_back(fb);
                    // con->setJointFeedback(fb);

                    m_dynamicsWorld->addConstraint(con,true);
                }
                prevBody = linkBody;

            }
            m_allbones_big_list.push_back(m_allbones);
            m_allhinges_big_list.push_back(m_allhinges);
            cout << "Push everything" << endl;
        }
        */
    }
	
	m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);
}

class CommonExampleInterface*    TestHingeTorqueCreateFunc(CommonExampleOptions& options){
	return new TestHingeTorque(options.m_guiHelper);
}
