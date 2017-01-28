cmake_minimum_required(VERSION 2.8)

find_package( Bullet REQUIRED )

FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

LINK_LIBRARIES(
    /usr/local/lib/libBullet3Common.${DYLIB_SUFFIX} 
)

IF (WIN32)
    ADD_EXECUTABLE(App_TestHinge
		TestHingeTorque.cpp
		main.cpp
		${BULLET_PHYSICS_SOURCE_DIR}/build3/bullet.rc
	)
ELSE()
    ADD_EXECUTABLE(App_TestHinge
		TestHingeTorque.cpp
		main.cpp
	)
ENDIF()

target_link_libraries(App_TestHinge ${BULLET_LIBRARIES})
TARGET_LINK_LIBRARIES(App_TestHinge ${Boost_LIBRARIES} )
