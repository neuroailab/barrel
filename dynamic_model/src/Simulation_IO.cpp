
#include "Simulation_IO.hpp"

void save_data(sim_data* data, H5std_string filename){

	std::cout << "Saving data..." << std::endl;
	H5std_string datasetname("data");
	const int	 RANK = 3;

	try{
		// Turn off the auto-printing when failure occurs so that we can
		// handle the errors appropriately
		H5::Exception::dontPrint();

		// Create a new file using the default property lists. 
		H5::H5File file(filename, H5F_ACC_TRUNC);
		std::cout << "- File created." << std::endl;

		addToDataset3D(data->M, "M", file);
		addToDataset3D(data->F, "F", file);
		addToDataset3D(data->X, "X", file);
		addToDataset3D(data->Y, "Y", file);
		addToDataset3D(data->Z, "Z", file);
		addToDataset1D(data->T, "T", file);

	}  // end of try block

    // catch failure caused by the H5File operations
    catch(H5::FileIException error){
		error.printError();
    }

    // catch failure caused by the DataSet operations
    catch(H5::DataSetIException error){
		error.printError();
    }
    std::cout << "- Done." << std::endl;
}


void addToDataset3D(std::vector<std::vector<std::vector<float>>> data, H5std_string dataname, H5::H5File file){

	const int	 RANK = 3;

	// Create the data space for the dataset.
	hsize_t dims[3];               // dataset dimensions

	dims[0] = data.size();		// get time dimension
	dims[1] = data[0].size();	// get whisker dimension
	dims[2] = data[0][0].size(); // get mechanical dimension

	int T_DIM = data.size();		// get time dimension
	int W_DIM = data[0].size();	// get whisker dimension
	int D_DIM = data[0][0].size(); // get data dimension

	dims[0] = T_DIM;  	// set time dimension
	dims[1] = W_DIM;	// set whisker dimension
	dims[2] = D_DIM;	// set data dimension
	


	// std::cout << "- Data set dimensions: " << dims[0] << ", " << dims[1] << ", " << dims[2] << std::endl;

	float data_array[T_DIM][W_DIM][D_DIM] = {0};
	// std::cout << "- Data array declared." << std::endl;
	for(int i=0; i<W_DIM; i++){
		for(int j=0; j<D_DIM; j++){
			for(int k=0; k<T_DIM; k++){
				data_array[k][i][j] = data[k][i][j]; // reorganize and convert data vectors to array
			}
		}
	}



	H5::DataSpace dataspace(RANK, dims);

	// Create the dataset    
	H5::DataSet dataset = file.createDataSet(dataname, H5::PredType::NATIVE_FLOAT, dataspace);
	dataset.write(data_array, H5::PredType::NATIVE_FLOAT);
	std::cout << "- Data set < " << dataname << " > attached." << std::endl;
}

void addToDataset1D(std::vector<float> data, H5std_string dataname, H5::H5File file){

	const int	 RANK = 1;

	// Create the data space for the dataset
	hsize_t dim[1];               // dataset dimensions

	dim[0] = data.size();		// get time dimension

	float data_array[dim[0]] = {0};
	for(int i=0; i<dim[0]; i++){
		data_array[i] = data[i]; // reorganize and convert data vectors to array
	}

	H5::DataSpace dataspace(RANK, dim);

	// Create the dataset.      
	H5::DataSet dataset = file.createDataSet(dataname, H5::PredType::NATIVE_FLOAT, dataspace);
	dataset.write(data_array, H5::PredType::NATIVE_FLOAT);
	std::cout << "- Data set < " << dataname << " > attached." << std::endl;
}