# App_Simulation is a minimal sample creating, stepping and deleting a Bullet dynamics world
cmake_minimum_required(VERSION 3.7)

SET(CMAKE_CXX_FLAGS "-std=c++11")

SET(BULLET_SOURCE_DIR ~/bullet3 CACHE STRING "Some other user-specified option")
SET(BULLET_BUILD_DIR ~/bullet3/bullet_build CACHE STRING "Some other user-specified option")

FIND_PACKAGE(HDF5 COMPONENTS C CXX REQUIRED)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
FIND_PACKAGE(OpenGL)

INCLUDE_DIRECTORIES(    
        include
        ${BULLET_SOURCE_DIR}/src
        ${BULLET_SOURCE_DIR}/Extras/Serialize/BulletWorldImporter
        /usr/local/include/hdf5/serial
        ${Boost_INCLUDE_DIR}
        
)


SET(SOURCES
    src/Parameters.cpp
    src/Simulation_utility.cpp
    src/Simulation_IO.cpp
    src/Whisker_utility.cpp
    src/Whisker_config.cpp
    src/Whisker.cpp
    src/WhiskerArray.cpp
    src/Simulation.cpp

)

SET(EXT_SOURCES
    ${BULLET_SOURCE_DIR}/build3/bullet.rc
    #${BULLET_SOURCE_DIR}/examples/StandaloneMain/main_opengl_single_example.cpp
    ${BULLET_SOURCE_DIR}/examples/ExampleBrowser/OpenGLGuiHelper.cpp
   ${BULLET_SOURCE_DIR}/examples/ExampleBrowser/GL_ShapeDrawer.cpp
    ${BULLET_SOURCE_DIR}/examples/ExampleBrowser/CollisionShape2TriangleMesh.cpp
    ${BULLET_SOURCE_DIR}/examples/Utils/b3Clock.cpp

)

LINK_LIBRARIES(
 BulletDynamics BulletCollision LinearMath
)

IF (WIN32)
	ADD_EXECUTABLE(App_Whisker
        ${EXT_SOURCES}
        ${SOURCES}
		${CMAKE_SOURCE_DIR}/src/main.cpp
		
	)
ELSE()
	ADD_EXECUTABLE(App_Whisker
        ${SOURCES}
        #${EXT_SOURCES}
        ${BULLET_SOURCE_DIR}/build3/bullet.rc 
		${CMAKE_SOURCE_DIR}/src/main.cpp
	)
ENDIF()

#TARGET_INCLUDE_DIRECTORIES(App_Whisker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(App_Whisker ${HDF5_CXX_LIBRARIES} ${Boost_LIBRARIES})

IF (INTERNAL_ADD_POSTFIX_EXECUTABLE_NAMES)
			SET_TARGET_PROPERTIES(App_Whisker PROPERTIES  DEBUG_POSTFIX "_Debug")
			SET_TARGET_PROPERTIES(App_Whisker PROPERTIES  MINSIZEREL_POSTFIX "_MinsizeRel")
			SET_TARGET_PROPERTIES(App_Whisker PROPERTIES  RELWITHDEBINFO_POSTFIX "_RelWithDebugInfo")
ENDIF(INTERNAL_ADD_POSTFIX_EXECUTABLE_NAMES)





#################
# Standalone SimulationGui using OpenGL (but not the example browser)


INCLUDE_DIRECTORIES(
        include
        ${BULLET_SOURCE_DIR}/src
        ${BULLET_SOURCE_DIR}/btgui
        ${BULLET_SOURCE_DIR}/examples
        ${BULLET_SOURCE_DIR}/examples/ThirdPartyLibs/Glew
        ${BULLET_SOURCE_DIR}/examples/OpenGLWindow
        ${BULLET_SOURCE_DIR}/Extras/Serialize/BulletWorldImporter
        /usr/local/include/hdf5/serial
        ${Boost_INCLUDE_DIR}
)

#LINK_DIRECTORIES( ${HDF5_LIBRARY_DIRS} )

#this define maps StandaloneExampleCreateFunc to the right 'CreateFunc'
ADD_DEFINITIONS(-DB3_USE_STANDALONE_EXAMPLE)

LINK_LIBRARIES(
        BulletDynamics BulletCollision LinearMath OpenGLWindow Bullet3Common 
)

#some code to support OpenGL and Glew cross platform
IF (WIN32)
        INCLUDE_DIRECTORIES(
             ${BULLET_SOURCE_DIR}/btgui/OpenGLWindow/GlewWindows
        )
        ADD_DEFINITIONS(-DGLEW_STATIC)
	LINK_LIBRARIES( ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} )
ELSE(WIN32)
        IF(APPLE)
                find_library(COCOA NAMES Cocoa)
                MESSAGE(${COCOA})
                link_libraries(${COCOA} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})

        ELSE(APPLE)
                INCLUDE_DIRECTORIES(
                        ${BULLET_SOURCE_DIR}/btgui/OpenGLWindow/GlewWindows
                )
                ADD_DEFINITIONS("-DGLEW_INIT_OPENGL11_FUNCTIONS=1")
                ADD_DEFINITIONS("-DGLEW_STATIC")
                ADD_DEFINITIONS("-DGLEW_DYNAMIC_LOAD_ALL_GLX_FUNCTIONS=1")

                LINK_LIBRARIES(  pthread ${DL} )
        ENDIF(APPLE)
ENDIF(WIN32)


ADD_EXECUTABLE(AppWhiskerGui
    ${EXT_SOURCES}
    ${SOURCES}
    ${CMAKE_SOURCE_DIR}/src/main_opengl.cpp
)

#TARGET_INCLUDE_DIRECTORIES(AppWhiskerGui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(AppWhiskerGui ${HDF5_CXX_LIBRARIES} ${Boost_LIBRARIES} ${BULLET_BUILD_DIR}/examples/OpenGLWindow/libOpenGLWindow.so)

IF (INTERNAL_ADD_POSTFIX_EXECUTABLE_NAMES)
                        SET_TARGET_PROPERTIES(AppWhiskerGui PROPERTIES  DEBUG_POSTFIX "_Debug")
                        SET_TARGET_PROPERTIES(AppWhiskerGui PROPERTIES  MINSIZEREL_POSTFIX "_MinsizeRel")
                        SET_TARGET_PROPERTIES(AppWhiskerGui PROPERTIES  RELWITHDEBINFO_POSTFIX "_RelWithDebugInfo")
ENDIF(INTERNAL_ADD_POSTFIX_EXECUTABLE_NAMES)


