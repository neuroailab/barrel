
#include "Whisker_utility.hpp"

// std::map<char,int> rownum = {{'A', 1}, {'B', 2}, {'C', 3}, {'D', 4}, {'E', 5}};
// std::map<char,int> colnum = {{'0', 0}, {'1', 1}, {'2', 2}, {'3', 3}, {'4', 4}, {'5', 5}, {'6', 6}, {'7', 7}};


float calc_x_from_a_and_s(float a, float s){
    
    //some strange helper function that calculates curvature
    float x = -1.4 + 133.7*a +1.27*s - 4334*pow(a,2) - 7.3*a*s - 0.01*s*2
            + 62930*pow(a,3) + 72.2*pow(a,2)*s - 0.2*a*pow(s,2) - 407600*pow(a,4)
            - 565.2*pow(a,3)*s + 1.6*pow(a,2)*pow(s,2) + 966000*pow(a,5) + 1706*pow(a,4)*s
            - 4.8*pow(a,3)*pow(s,2);
    return x;
}



std::vector<float> get_angles_from_curvature(float L, float a, int numLinks){
    /*Gets the nodal angles based on the whisker's curvature.
        a is the constant defining the curvature, y = a*x**2
        L is an array of link lengths*/
	// std::cout << "Calculating angles: " << std::endl;
    // in this code 'a' is a optimization parameter
    // Get the angles based on a whisker with parabolic shape.
	float linkLength = L/numLinks;
    std::vector<float> angle;
    float x0 = 0.;
    float y0 = 0.;
    float s, x, y, dx, dy;

    for(int l=0; l<numLinks; l++){
        s = x0 + linkLength;
        x = calc_x_from_a_and_s(a/1000.,s*1000.)/1000.; // so obnoxious that this works
        y = a*pow(x,2);
        dx = x - x0;
        dy = y - y0;
        x0 = x;
        y0 = y;
        angle.push_back(-atan(dy/dx));
    }

    for(int i=0; i<angle.size(); i++){
    	angle[i] = angle[i] - angle[0];
    }    

    // Now adjust angles to be relative to one another
    int n = 0;
    for (std::vector<float>::iterator it=angle.begin(); it != angle.end() ;++it)//iterating thru each elementn in vector 
    {	
    	float cumsum = std::accumulate(angle.begin(), it, 0.0);
    	// std::cout << "- cumangle " << n << ": " << cumsum << std::endl;
        *it= *it - cumsum;
        // std::cout << "- angle " << n << ": " << *it << std::endl;
        n++;
    }

    return angle;
}

float calc_base_radius(int row, int col){
    float sTotal = 43 + 1.8*row - 7.6*col;
    float dBase = 0.041 + 0.002*sTotal + 0.011*row - 0.0039*col;
    return (dBase*1e-3)/2;
}
    
std::vector<float> calc_unit_radius(int numUnits, float rbase, float rtip){

    
    float r_unit;
	std::vector<float> r;
	for(int i=0; i<=numUnits; i++){
		r_unit = rbase + float(i)*(rtip-rbase)/float(numUnits+1);
		// std::cout << "Radius (unit " << i << ": " << r_unit << std::endl;
		r.push_back(r_unit);
	}
    return r;
}


std::vector<float> calc_mass(int numUnits, float unit_length, float rbase, float rtip, float rho){

    
    float m_unit;
	std::vector<float> m;
	for(int i=0; i<numUnits; i++){
        float rb = rbase + float(i)*(rtip-rbase)/float(numUnits);
        float rt = rbase + float(i+1)*(rtip-rbase)/float(numUnits);
		// std::cout << "Mass (unit " << i << ": " << m_unit << std::endl;
        m_unit = rho*(PI*unit_length/3)*(pow(rb,2) + rb*rt + pow(rt,2));
		m.push_back(m_unit);
	}
    return m;
}

std::vector<float> calc_inertia(int numUnits, float rbase, float rtip){
	float I_unit;
	std::vector<float> I;
	for(int i=0; i<numUnits; i++){
		I_unit = 0.25*PI*pow(rbase + float(i)*(rtip-rbase)/float(numUnits),4);
		// std::cout << "Inertia (unit " << i << ": " << I_unit << std::endl;
		I.push_back(I_unit);
	}
    return I;
}


std::vector<float> calc_young_modulus(int numUnits, float E_base, float E_tip){
    // Calculates damping coefficients at each node from a given damping ratio.
	float E_unit;
	std::vector<float> E;
	for(int i=0; i<numUnits; i++){
		E_unit = float(E_base + float(i)*(E_tip-E_base)/float(numUnits))*1e9;
		// std::cout << "Young Modulus (unit " << i << ": " << E_unit << std::endl;
		E.push_back(E_unit);
	}
   
    return E;
}

std::vector<float> calc_com(float L, float N, float rbase, float taper){
    // The location of the center of mass for each link
    // R = np.linspace(rbase, rbase*taper, N+1)
    // # the geometric centroid is calculated based on the formula for a frostum
    // return (L/4.)*(R[:-1]**2 + 2*R[:-1]*R[1:] + 3*R[1:]**2)/(R[:-1]**2 + R[:-1]*R[1:] + R[1:]**2)
}

std::vector<float> calc_stiffness(int numLinks, std::vector<float> E, std::vector<float> I, float unit_length){

    float k_unit;
	std::vector<float> k;
	for(int i=0; i<numLinks; i++){
		k_unit = E[i]*I[i]/unit_length*pow(SCALE,3);
		// std::cout << "Stiffness (unit " << i << ": " << k_unit << std::endl;
		k.push_back(k_unit);
	}
   
    return k;
}

std::vector<float> calc_damping(int numLinks, std::vector<float> k, std::vector<float> mass, float CoM, float zeta_base, float zeta_tip){
    

    float c_unit;
	std::vector<float> c;
	for(int i=0; i<numLinks; i++){
		c_unit = 2 * CoM * sqrt(k[i]*mass[i]) * (zeta_base + float(i)*(zeta_tip-zeta_base)/float(numLinks))*pow(SCALE,2)*20;
		// std::cout << "Damping (unit " << i << ": " << c_unit << std::endl;
		c.push_back(c_unit);
	}
   
    return c;
}



/**
 * == How To Use ERP and CFM ==
 * ERP and CFM can be independently set in many joints. They can be set in contact joints, in joint limits and various other places, to control the spongyness and springyness of the joint (or joint limit).
 * If CFM is set to zero, the constraint will be hard. If CFM is set to a positive value, it will be possible to violate the constraint by "pushing on it" (for example, for contact constraints by forcing the two contacting objects together). In other words the constraint will be soft, and the softness will increase as CFM increases. What is actually happening here is that the constraint is allowed to be violated by an amount proportional to CFM times the restoring force that is needed to enforce the constraint. Note that setting CFM to a negative value can have undesirable bad effects, such as instability. Don't do it.
 * By adjusting the values of ERP and CFM, you can achieve various effects. For example you can simulate springy constraints, where the two bodies oscillate as though connected by springs. Or you can simulate more spongy constraints, without the oscillation. In fact, ERP and CFM can be selected to have the same effect as any desired spring and damper constants. If you have a spring constant k_p and damping constant k_d, then the corresponding ODE constants are:
 *
 * ERP = \frac{h k_p} {h k_p + k_d}
 *
 * CFM = \frac{1} {h k_p + k_d}
 *
 * where h is the step size. These values will give the same effect as a spring-and-damper system simulated with implicit first order integration.
 * Increasing CFM, especially the global CFM, can reduce the numerical errors in the simulation. If the system is near-singular, then this can markedly increase stability. In fact, if the system is misbehaving, one of the first things to try is to increase the global CFM.
 * @link http://ode-wiki.org/wiki/index.php?title=Manual:_All&printable=yes#How_To_Use_ERP_and_CFM
 * @return
 */

/**
 * Joint error and the Error Reduction Parameter (ERP)
 *
 * When a joint attaches two bodies, those bodies are required to have certain positions and orientations relative to each other. However, it is possible for the bodies to be in positions where the joint constraints are not met. This "joint error" can happen in two ways:
 *
 * If the user sets the position/orientation of one body without correctly setting the position/orientation of the other body.
 * During the simulation, errors can creep in that result in the bodies drifting away from their required positions.
 * Figure 3  shows an example of error in a ball and socket joint (where the ball and socket do not line up).
 *
 * There is a mechanism to reduce joint error: during each simulation step each joint applies a special force to bring its bodies back into correct alignment. This force is controlled by the error reduction parameter (ERP), which has a value between 0 and 1.
 *
 * The ERP specifies what proportion of the joint error will be fixed during the next simulation step. If ERP = 0 then no correcting force is applied and the bodies will eventually drift apart as the simulation proceeds. If ERP=1 then the simulation will attempt to fix all joint error during the next time step. However, setting ERP=1 is not recommended, as the joint error will not be completely fixed due to various internal approximations. A value of ERP=0.1 to 0.8 is recommended (0.2 is the default).
 *
 * A global ERP value can be set that affects most joints in the simulation. However some joints have local ERP values that control various aspects of the joint.
 * @link http://ode-wiki.org/wiki/index.php?title=Manual:_All&printable=yes#How_To_Use_ERP_and_CFM
 * @return
 */

btScalar getERP(btScalar timeStep, btScalar k, btScalar c) {
	return timeStep * k / (timeStep * k + c);
}

/**
 * Most constraints are by nature "hard". This means that the constraints represent conditions that are never violated. For example, the ball must always be in the socket, and the two parts of the hinge must always be lined up. In practice constraints can be violated by unintentional introduction of errors into the system, but the error reduction parameter can be set to correct these errors.
 * Not all constraints are hard. Some "soft" constraints are designed to be violated. For example, the contact constraint that prevents colliding objects from penetrating is hard by default, so it acts as though the colliding surfaces are made of steel. But it can be made into a soft constraint to simulate softer materials, thereby allowing some natural penetration of the two objects when they are forced together.
 * There are two parameters that control the distinction between hard and soft constraints. The first is the error reduction parameter (ERP) that has already been introduced. The second is the constraint force mixing (CFM) value, that is described below.
 *
 * == Constraint Force Mixing (CFM) ==
 * What follows is a somewhat technical description of the meaning of CFM. If you just want to know how it is used in practice then skip to the next section.
 * Traditionally the constraint equation for every joint has the form
 * \mathbf{J} v = \mathbf{c}
 * where v is a velocity vector for the bodies involved, \mathbf{J} is a "Jacobian" matrix with one row for every degree of freedom the joint removes from the system, and \mathbf{c} is a right hand side vector. At the next time step, a vector \lambda is calculated (of the same size as \mathbf{c}) such that the forces applied to the bodies to preserve the joint constraint are:
 * F_c = \mathbf{J}^T \lambda
 * ODE adds a new twist. ODE's constraint equation has the form
 *
 * \mathbf{J} v = \mathbf{c} + \textbf{CFM} \, \lambda
 * where CFM is a square diagonal matrix. CFM mixes the resulting constraint force in with the constraint that produces it. A nonzero (positive) value of CFM allows the original constraint equation to be violated by an amount proportional to CFM times the restoring force \lambda that is needed to enforce the constraint. Solving for \lambda gives
 *
 * (\mathbf{J} \mathbf{M}^{-1} \mathbf{J}^T + \frac{1}{h}  \textbf{CFM}) \lambda = \frac{1}{h} \mathbf{c}
 * Thus CFM simply adds to the diagonal of the original system matrix. Using a positive value of CFM has the additional benefit of taking the system away from any singularity and thus improving the factorizer accuracy.
 * @link http://ode-wiki.org/wiki/index.php?title=Manual:_All&printable=yes#How_To_Use_ERP_and_CFM
 * @return
 */
btScalar getCFM(btScalar avoidSingularity, btScalar timeStep, btScalar k,btScalar c) {
	return btScalar(avoidSingularity) / (timeStep * k + c);
}