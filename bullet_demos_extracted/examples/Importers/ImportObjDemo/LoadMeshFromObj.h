#ifndef LOAD_MESH_FROM_OBJ_H
#define LOAD_MESH_FROM_OBJ_H


struct GLInstanceGraphicsShape;
class btCollisionShape;

GLInstanceGraphicsShape* LoadMeshFromObj(const char* relativeFileName, const char* materialPrefixPath);
btCollisionShape* LoadShapesFromObj(const char* relativeFileName, const char* materialPrefixPath);

#endif //LOAD_MESH_FROM_OBJ_H

