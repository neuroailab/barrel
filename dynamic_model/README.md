# whiskerBullet


In this repository, we're building a simulation tool for active whisking based on the open source Bullet physics engine and OpenGL. 

## Installation Instructions:

1. Clone bullet from https://github.com/bulletphysics/bullet3

2. Follow installation instructions on http://bulletphysics.org/mediawiki-1.5.8/index.php/Installation or on Linux do:
```
	cd path/to/bullet
	mkdir bullet_build
	cd bullet_build
	cmake .. -G "Unix Makefiles" -DINSTALL_LIBS=ON -DBUILD_SHARED_LIBS=ON
	make -j4
	sudo make install
```
3. Install Boost 1.62 library with `sudo apt-get install libboost1.62-all-dev`

4. Install Hdf5 library with `sudo apt-get install libhdf5-cpp-100`

5. Clone whiskerBullet

6. Compile whiskerBullet with:
```
	cd your/path/to/whiskerBullet
	mkdir build
	cd build
	cmake -DBULLET_SOURCE_DIR:STRING=/your/path/to/bullet/source/directory -DBULLET_BUILD_DIR:STRING=/your/path/to/bullet/build/directory ..
	make
```
7. Run `App_Whisker` (no graphics) or `AppWhiskerGui` (with graphics). Use --help or -h for information about command line arguments
 
