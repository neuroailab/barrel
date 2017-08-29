# whiskerBullet


In this repository, we're building a simulation tool for active whisking based on the open source Bullet physics engine and OpenGL (Freeglut3). 

## Installation Instructions:

1. Install Freeglut with `sudo apt-get install freeglut3 freeglut3-dev`

2. Clone bullet from https://github.com/bulletphysics/bullet3

3. Follow installation instructions on http://bulletphysics.org/mediawiki-1.5.8/index.php/Installation or on Linux do:
```
	cd path/to/bullet
	mkdir bullet-build
	cd bullet-build
	cmake .. -G "Unix Makefiles" -DINSTALL_LIBS=ON -DBUILD_SHARED_LIBS=ON
	make -j4
	sudo make install
```
4. Install Boost 1.62 library with `sudo apt-get install libboost1.62-all-dev`

5. Install Hdf5 library with `sudo apt-get install libhdf5-cpp-100`

4. Clone WhiskerBullet

5. Rename CMakeLists.txt in `your/path/to/bullet/examples` for backup.

6. Copy the folder `Whiskers` and the file `CMakeLists.txt` into `your/path/to/bullet/examples`.

7. Recompile bullet with:
```
	cd your/path/to/bullet/bullet-build
	cmake ..
	make
```
8. Run `App_Whisker` (no graphics) or `AppWhiskerGui` (with graphics) in `your/path/to/bullet/bullet-build/examples/Whiskers`.
 
