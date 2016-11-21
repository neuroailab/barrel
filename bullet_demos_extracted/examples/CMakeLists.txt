cmake_minimum_required( VERSION 2.8 )
SET(BULLET_VERSION 2.85)
SET(BULLET_PHYSICS_SOURCE_DIR /Users/chengxuz/barrel/bullet/bullet3/)

find_package( Bullet REQUIRED )
include_directories(${BULLET_INCLUDE_DIRS})
SUBDIRS( HelloWorld BasicDemo )
SUBDIRS( ExampleBrowser ThirdPartyLibs/Gwen ThirdPartyLibs/BussIK OpenGLWindow )
