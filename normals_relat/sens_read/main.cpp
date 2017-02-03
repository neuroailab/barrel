

#include <stdio.h>
#include <stdlib.h>

#include <stdio.h>

#if _WIN32
#include "windows.h"
#include <tchar.h>
#endif 

#include "sensorData.h"



int main(int argc, char* argv[])
{
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	//_CrtSetBreakAlloc(7545);
	try {
		//non-cached read
		//const std::string filename = "test.sens";
		const std::string filename = "2016-08-06_02-59-01__C7BA9586-8237-4204-9116-02AE5804338A.sens";
		ml::SensorData sd;

		std::cout << "loading from file... ";
		sd.loadFromFile(filename);
		std::cout << "done!" << std::endl;

		std::cout << sd << std::endl;

		for (size_t i = 0; i < sd.m_frames.size(); i++) {
			std::cout << "\r[ processing frame " << std::to_string(i) << " of " << std::to_string(sd.m_frames.size()) << " ]";
			ml::vec3uc* colorData = sd.decompressColorAlloc(i);
			unsigned short* depthData = sd.decompressDepthAlloc(i);

			sd.m_colorWidth;
			sd.m_colorHeight;
			sd.m_depthWidth;
			sd.m_depthHeight;

			//convert to m:
			float depth_in_meters = sd.m_depthShift * depthData[0];


			std::free(colorData);
			std::free(depthData);
		}

		//const std::string out_name = "test.unkn";
		//sd.saveToFile(out_name);
		const std::string out_name = "testImages/";
		//sd.saveToImages();
		std::cout << std::endl;
	}
	catch (const std::exception& e)
	{
		//MessageBoxA(NULL, e.what(), "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	catch (...)
	{
		//MessageBoxA(NULL, "UNKNOWN EXCEPTION", "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	
	std::cout << "<press key to continue>" << std::endl;
	getchar();
	return 0;
}

