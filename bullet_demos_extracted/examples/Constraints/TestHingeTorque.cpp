#include "TestHingeTorque.h"
#include <boost/program_options.hpp>

#include <iostream>
#include <iterator>
#include <fstream>

namespace po = boost::program_options;
using namespace std;

#include "../CommonInterfaces/CommonRigidBodyBase.h"
#include "../CommonInterfaces/CommonParameterInterface.h"

short collisionFilterGroup = short(btBroadphaseProxy::CharacterFilter);
short collisionFilterMask = short(btBroadphaseProxy::AllFilter ^ (btBroadphaseProxy::CharacterFilter));
static btScalar radius(0.2);

//        btVector3 linkHalfExtents(0.05, 0.37, 0.1);
//const float x_len_link  = 0.05;
//const float x_len_link  = 0.03;
float x_len_link  = 0.53;
//const float y_len_link  = 0.37;
//const float y_len_link  = 0.18;
float y_len_link  = 2.08;
//const float y_len_link  = 0.15;
//const float z_len_link  = 0.1;
float z_len_link  = 0.3;
//const float basic_str   = 2000;
float basic_str   = 3000;
//const int const_numLinks = 8;
//const int const_numLinks = 8;
int const_numLinks = 15;
//const int const_numLinks = 25;
//const int const_numLinks = 2;
//const int const_numLinks = 2;
//const float linear_damp = 0.7;
//const float ang_damp = 0.9;
//const float linear_damp = 0.37;
//const float ang_damp = 0.39;
float linear_damp = 0.77;
float ang_damp = 0.79;
//const float linear_damp = 0.07;
//const float ang_damp = 0.09;
float time_leap = 1.0/240.0;
//const int limit_numLink = const_numLinks - 1;
int limit_numLink = const_numLinks + 1;
//const float equi_angle = -0.1;
float equi_angle = 0;
//const int inter_spring = 1;
int inter_spring = 5;
//const int every_spring = 5;
//const int every_spring = 8;
//const int every_spring = 10;
int every_spring = 1;
//const float spring_stiffness = 220; //Works for 8
//const float spring_stiffness = 420; //Tmp for 15
float spring_stiffness = 520;
float camera_dist     = 45;
//const float linear_damp = 0.95;
//const float ang_damp = 0.9;
//const float spring_offset   = 0.5;
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
//int debug_flag          = 

struct TestHingeTorque : public CommonRigidBodyBase
{
    bool m_once;
    float pass_time;
    float curr_velo;
    float curr_angl;
    float curr_state;
    float curr_force;
    float curr_torque;
    btAlignedObjectArray<btJointFeedback*> m_jointFeedback;
    btAlignedObjectArray<btRigidBody*> m_allbones;
    btAlignedObjectArray<btHingeConstraint*> m_allhinges;

	TestHingeTorque(struct GUIHelperInterface* helper);
	virtual ~ TestHingeTorque();
	virtual void initPhysics();

	virtual void stepSimulation(float deltaTime);

	
	virtual void resetCamera()
	{
    
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
m_once(true)
{
}
TestHingeTorque::~ TestHingeTorque()
{
	for (int i=0;i<m_jointFeedback.size();i++)
	{
		delete m_jointFeedback[i];
	}

}


void TestHingeTorque::stepSimulation(float deltaTime)
{
    if (0)//m_once)
    {
        m_once=false;
        btHingeConstraint* hinge = (btHingeConstraint*)m_dynamicsWorld->getConstraint(0);
        
        btRigidBody& bodyA = hinge->getRigidBodyA();
        btTransform trA = bodyA.getWorldTransform();
        btVector3 hingeAxisInWorld = trA.getBasis()*hinge->getFrameOffsetA().getBasis().getColumn(2);
        hinge->getRigidBodyA().applyTorque(-hingeAxisInWorld*10);
        hinge->getRigidBodyB().applyTorque(hingeAxisInWorld*10);
        
    }
    
    m_dynamicsWorld->stepSimulation(time_leap,0);

    pass_time   = pass_time + time_leap;

    if (flag_time==1){
        if (pass_time > time_limit){
            exit(0);
        }
    }
	
    curr_velo   = 0;
    curr_angl   = 0;
    curr_state  = 0;
    curr_force  = 0;
    curr_torque = 0;
    static int count = 0;
    //if ((count& 0x0f)==0)
    if (1)
    {
        int all_size    = m_allbones.size();
        btRigidBody* base = btRigidBody::upcast(m_dynamicsWorld->getCollisionObjectArray()[0]);

        for (int i=0;i<all_size;i++){
            btVector3 tmp_center_pos = m_allbones[i]->getCenterOfMassPosition();

            btTransform tmp_trans = m_allbones[i]->getCenterOfMassTransform();
            btVector3 tmp_pos = tmp_trans(btVector3(0,0,-z_len_link));
            btVector3 tmp_direction = tmp_center_pos - tmp_pos;
            btVector3 tmp_pos_f = tmp_trans(btVector3(0,y_len_link,0));
            btVector3 tmp_point     = tmp_pos_f - tmp_center_pos;

            btScalar tmp_angle_high     = m_allhinges[i]->getHingeAngle() - equi_angle;

            btVector3 tmp_force_now     = tmp_direction*tmp_angle_high*basic_str*(m_allbones.size() - i)/2;
            m_allbones[i]->applyForce(tmp_force_now, -tmp_point);

            if (i < all_size -1){
                btTransform tmp_trans = m_allbones[i+1]->getCenterOfMassTransform();
                btVector3 tmp_pos = tmp_trans(btVector3(0,0,z_len_link));
                btVector3 tmp_pos_f = tmp_trans(btVector3(0,-y_len_link,0));
                btVector3 tmp_point     = tmp_pos_f - tmp_center_pos;
                m_allbones[i+1]->applyForce(-tmp_force_now, -tmp_point);
            }

            if ((i==initial_poi) && (pass_time < initial_stime)){
                m_allbones[i]->applyForce(btVector3(0,0,initial_str), btVector3(0,0,0));
            }

        }

        for (int i=0;i<all_size;i++){
            curr_velo   += m_allbones[i]->getLinearVelocity().norm();
            curr_angl   += m_allbones[i]->getAngularVelocity().norm();
            curr_force  += m_allbones[i]->getTotalForce().norm();
            curr_torque += m_allbones[i]->getTotalTorque().norm();
        }

        curr_velo   /= all_size;
        curr_angl   /= all_size;
        curr_force  /= all_size;
        curr_torque /= all_size;

        for (int i=0;i<m_allhinges.size();i++){
            curr_state  += m_allhinges[i]->getHingeAngle() - limit_up;
        }
        curr_state  /= m_allhinges.size();
    }
    count++;

    if (flag_time==2){
        if ((curr_velo < velo_ban_limit) && (curr_state < state_ban_limit) && (curr_angl < angl_ban_limit) && (curr_force < force_limit) && (curr_torque < torque_limit)){
            exit(0);
        }
        cout << "Now state:" << curr_velo << " " << curr_angl << " " << curr_force << " " << curr_torque << " " << curr_state << endl;
    }

    //CommonRigidBodyBase::stepSimulation(deltaTime);
}



void TestHingeTorque::initPhysics()
{
	int upAxis = 1;
    pass_time   = 0;
    m_allbones.clear();
    m_allhinges.clear();
    
    //b3Printf("Config name = %s",m_guiHelper->m_data->m_glApp->configname);
    //b3Printf("Config name = %s",m_guiHelper->configname);
    b3Printf("Config name = %s",m_guiHelper->getconfigname());

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("x_len_link", po::value<float>(), "Size x of cubes")
            ("y_len_link", po::value<float>(), "Size y of cubes")
            ("z_len_link", po::value<float>(), "Size z of cubes")
            ("basic_str", po::value<float>(), "Minimal strength of hinge's recover force")
            ("const_numLinks", po::value<int>(), "Number of units")
            ("linear_damp", po::value<float>(), "Control the linear damp ratio")
            ("ang_damp", po::value<float>(), "Control the angle damp ratio")
            ("time_leap", po::value<float>(), "Time unit for simulation")
            ("equi_angle", po::value<float>(), "Control the angle of balance for hinges")
            ("inter_spring", po::value<int>(), "Number of units between two strings")
            ("every_spring", po::value<int>(), "Number of units between one strings")
            ("spring_stiffness", po::value<float>(), "Stiffness of spring")
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
        ;

        po::variables_map vm;
        const char* file_name = m_guiHelper->getconfigname();
        ifstream ifs(file_name);
        po::store(po::parse_config_file(ifs, desc), vm);
        po::notify(vm);    

        if (vm.count("x_len_link")){
            x_len_link      = vm["x_len_link"].as<float>();
        }
        if (vm.count("y_len_link")){
            y_len_link      = vm["y_len_link"].as<float>();
        }
        if (vm.count("z_len_link")){
            z_len_link      = vm["z_len_link"].as<float>();
        }

        if (vm.count("basic_str")){
            basic_str       = vm["basic_str"].as<float>();
        }

        if (vm.count("const_numLinks")){
            const_numLinks  = vm["const_numLinks"].as<int>();
        }
        limit_numLink = const_numLinks + 1;

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
            inter_spring    = vm["inter_spring"].as<int>();
        }
        if (vm.count("every_spring")){
            every_spring    = vm["every_spring"].as<int>();
        }

        if (vm.count("spring_stiffness")){
            spring_stiffness    = vm["spring_stiffness"].as<float>();
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
	
        //m_dynamicsWorld->setGravity(btVector3(0,0,-10));
    m_dynamicsWorld->setGravity(btVector3(0,0,0));
    
	m_guiHelper->createPhysicsDebugDrawer(m_dynamicsWorld);
	int mode = 	btIDebugDraw::DBG_DrawWireframe
				+btIDebugDraw::DBG_DrawConstraints
				+btIDebugDraw::DBG_DrawConstraintLimits;
	m_dynamicsWorld->getDebugDrawer()->setDebugMode(mode);


	{ // create a door using hinge constraint attached to the world
        
        int numLinks = const_numLinks;
        bool spherical = false;					//set it ot false -to use 1DoF hinges instead of 3DoF sphericals
        bool canSleep = false;
        bool selfCollide = false;
        btVector3 linkHalfExtents(x_len_link, y_len_link, z_len_link);
        btVector3 baseHalfExtents(x_len_link, y_len_link, z_len_link);

        btBoxShape* baseBox = new btBoxShape(baseHalfExtents);
        //btVector3 basePosition = btVector3(-0.4f, 3.f, 0.f);
        btVector3 basePosition = btVector3(-0.4f, 7.f, 0.f);
        btTransform baseWorldTrans;
        baseWorldTrans.setIdentity();
        baseWorldTrans.setOrigin(basePosition);
        
        //mbC->forceMultiDof();							//if !spherical, you can comment this line to check the 1DoF algorithm
        //init the base
        btVector3 baseInertiaDiag(0.f, 0.f, 0.f);
        float baseMass = 0.f;
        float linkMass = 1.f;
        
        btRigidBody* base = createRigidBody(baseMass,baseWorldTrans,baseBox);
        m_dynamicsWorld->removeRigidBody(base);
        base->setDamping(0,0);
        m_dynamicsWorld->addRigidBody(base,collisionFilterGroup,collisionFilterMask);
        btBoxShape* linkBox1 = new btBoxShape(linkHalfExtents);
		btSphereShape* linkSphere = new btSphereShape(radius);
		
        btRigidBody* prevBody = base;
        
        for (int i=0;i<numLinks;i++)
        {
            btTransform linkTrans;
            linkTrans = baseWorldTrans;
            
            linkTrans.setOrigin(basePosition-btVector3(0,linkHalfExtents[1]*2.f*(i+1),0));
            
			btCollisionShape* colOb = 0;
			
			if (i<limit_numLink)
			{
				colOb = linkBox1;
			} else 
			{
				colOb = linkSphere;
			}
            btRigidBody* linkBody = createRigidBody(linkMass,linkTrans,colOb);
            m_dynamicsWorld->removeRigidBody(linkBody);
            //linkBody->setDamping(0,0);
            linkBody->setDamping(linear_damp,ang_damp);
            //linkBody->m_additionalDamping   = true;
            m_dynamicsWorld->addRigidBody(linkBody,collisionFilterGroup,collisionFilterMask);
            if (i<limit_numLink){
                m_allbones.push_back(linkBody);
            }
			btTypedConstraint* con = 0;
			
			if (i<limit_numLink)
			{
				//create a hinge constraint
				btVector3 pivotInA(0,-linkHalfExtents[1],0);
				btVector3 pivotInB(0,linkHalfExtents[1],0);
				btVector3 axisInA(1,0,0);
				btVector3 axisInB(1,0,0);
				bool useReferenceA = true;
				btHingeConstraint* hinge = new btHingeConstraint(*prevBody,*linkBody,
                                                                                 pivotInA,pivotInB,
                                                                                 axisInA,axisInB,useReferenceA);
                                //hinge->setLimit(-1.f, 1.f, 100.9f, 1.3f, 1.0f);
                                //hinge->setLimit(-2.f, 2.f);
                                //
                hinge->setLimit(limit_low, limit_up, limit_softness, limit_bias, limit_relax);
                m_allhinges.push_back(hinge);
				con = hinge;

                                if ((i>every_spring-2) && (i % inter_spring==0)){
                                    //btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0, -linkHalfExtents[1], 0));
                                    //btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0, linkHalfExtents[1], 0));
                                    btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0, -every_spring*linkHalfExtents[1]+spring_offset, 0));
                                    btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0, every_spring*linkHalfExtents[1]-spring_offset, 0));
                                    btGeneric6DofSpring2Constraint* fixed;

                                    if (i-every_spring>-1) {
                                        fixed = new btGeneric6DofSpring2Constraint(*m_allbones[i-(every_spring)], *linkBody,pivotInA,pivotInB);
                                    } else {
                                        fixed = new btGeneric6DofSpring2Constraint(*base, *linkBody,pivotInA,pivotInB);
                                    }
                                    //for (int indx_tmp=0;indx_tmp<3;indx_tmp++){
                                    for (int indx_tmp=0;indx_tmp<6;indx_tmp++){
                                        fixed->enableSpring(indx_tmp, true);
                                        //fixed->setStiffness(indx_tmp, 10);
                                        fixed->setStiffness(indx_tmp, spring_stiffness);
                                    }
                                    fixed->setEquilibriumPoint();
                                    m_dynamicsWorld->addConstraint(fixed,true);
                                }
			} else
			{
				
				btTransform pivotInA(btQuaternion::getIdentity(),btVector3(0, -radius, 0));						//par body's COM to cur body's COM offset
				btTransform pivotInB(btQuaternion::getIdentity(),btVector3(0, radius, 0));							//cur body's COM to cur body's PIV offset
				btGeneric6DofSpring2Constraint* fixed = new btGeneric6DofSpring2Constraint(*prevBody, *linkBody,pivotInA,pivotInB);
				//fixed->setLinearLowerLimit(btVector3(0,0,0));
				//fixed->setLinearUpperLimit(btVector3(0,0,0));
				//fixed->setAngularLowerLimit(btVector3(0,0,0));
				//fixed->setAngularUpperLimit(btVector3(0,0,0));
				
				con = fixed;

			}
			btAssert(con);
			if (con)
			{
				btJointFeedback* fb = new btJointFeedback();
				m_jointFeedback.push_back(fb);
				con->setJointFeedback(fb);

				m_dynamicsWorld->addConstraint(con,true);
			}
			prevBody = linkBody;
            
        }
       
	}
	
	if (0)
	{
		btVector3 groundHalfExtents(1,1,0.2);
		groundHalfExtents[upAxis]=1.f;
		btBoxShape* box = new btBoxShape(groundHalfExtents);
		box->initializePolyhedralFeatures();
		
		btTransform start; start.setIdentity();
		btVector3 groundOrigin(-0.4f, 3.f, 0.f);
		btVector3 basePosition = btVector3(-0.4f, 3.f, 0.f);
		btQuaternion groundOrn(btVector3(0,1,0),0.25*SIMD_PI);
		
		groundOrigin[upAxis] -=.5;
		groundOrigin[2]-=0.6;
		start.setOrigin(groundOrigin);
	//	start.setRotation(groundOrn);
		btRigidBody* body =  createRigidBody(0,start,box);
		body->setFriction(0);
		
	}
	m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);
}

class CommonExampleInterface*    TestHingeTorqueCreateFunc(CommonExampleOptions& options)
{
	return new TestHingeTorque(options.m_guiHelper);
}
