

#include "Whisker_config.hpp"

// these are constants for whisker rotation
std::vector<float> dphi = {0.398,0.591,0.578,0.393,0.217};
std::vector<float> dzeta = {-0.9,-0.284,0.243,0.449, 0.744};


// function to obtain parameters for specific whisker
whisker_config get_parameters(std::string wname){

    // read in parameter file
    H5std_string filename("../data/whisker_param.h5");
    H5std_string groupname(wname);

    H5::DataSet wpos;
    H5::DataSet geom;
    H5::DataSet bp_coor;
    H5::DataSet bp_angles;

    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group group(file.openGroup(groupname));

    try {  // to determine if the dataset exists in the group
        wpos = H5::DataSet(group.openDataSet("wpos"));
        geom = H5::DataSet(group.openDataSet("geom"));
        bp_coor = H5::DataSet(group.openDataSet("BP_coor"));
        bp_angles = H5::DataSet(group.openDataSet("BP_angles"));
    }
    catch(H5::GroupIException not_found_error ) {
        std::cout << " Dataset for whisker parameters not found." << std::endl;
    }
    // std::cout << "Dataset open." << std::endl;

    int whisker_pos[3];
    double whisker_geom[2];
    double whisker_bp_coor[3];
    double whisker_bp_angles[3];

    wpos.read(whisker_pos, H5::PredType::NATIVE_INT);
    geom.read(whisker_geom, H5::PredType::NATIVE_DOUBLE);
    bp_coor.read(whisker_bp_coor, H5::PredType::NATIVE_DOUBLE);
    bp_angles.read(whisker_bp_angles, H5::PredType::NATIVE_DOUBLE);

    whisker_config wc;

    wc.id = wname;
    wc.side = whisker_pos[0];
    wc.row = whisker_pos[1];
    wc.col = whisker_pos[2];
    wc.L = whisker_geom[0]/1000.;
    wc.a = whisker_geom[1]*1000.;

    wc.base_pos = btVector3(whisker_bp_coor[0],whisker_bp_coor[1],whisker_bp_coor[2])/1000;
    wc.base_rot = btVector3(whisker_bp_angles[0],whisker_bp_angles[1],whisker_bp_angles[2]);

    return wc;
}

// function to get zeta angle of whisker motion (depends on row)
float get_dzeta(int index){

	return dzeta[index];
}

// function to get phi angle of whisker motion (depends on row)
float get_dphi(int index){

	return dphi[index];
}


