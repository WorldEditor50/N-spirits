#ifndef MESH_HPP
#define MESH_HPP
#include <vector>
#include "../basic/point.hpp"

class Mesh
{
public:
    std::vector<Point3d> p1;
    std::vector<Point3d> p2;
    std::vector<Point3d> p3;
public:
    Mesh(){}
};

#endif // MESH_HPP
