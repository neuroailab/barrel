cmake_minimum_required( VERSION 2.8 )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(BULLET_VERSION 2.85)
SET(BULLET_PHYSICS_SOURCE_DIR /Users/chengxuz/barrel/bullet/bullet3/)

find_package( Bullet REQUIRED )
include_directories(${BULLET_INCLUDE_DIRS})
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

SUBDIRS( HelloWorld BasicDemo )
SUBDIRS( ExampleBrowser ThirdPartyLibs/Gwen ThirdPartyLibs/BussIK OpenGLWindow )
