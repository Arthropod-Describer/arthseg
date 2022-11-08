#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/ndarraytypes.h>
#include <vector>

#include "types.hpp"

struct Component
{
    size_t label;
    std::vector<Point> nodes;

    Component(const int label) : label(label) {}
    Component(const int label, const Point &point) : label(label), nodes({ point }) {}
    bool empty() const { return nodes.empty(); }
    size_t size() const { return nodes.size(); }
    bool operator<(const Component &other) const
    {
        return label < other.label; // || (label == other.label && size() >= other.size());
    }
};

struct ComponentWithEdge : Component
{
    std::vector<Point> edge;
    ComponentWithEdge(int label) : Component(label) {}
    ComponentWithEdge(const int label, const Point &point) : Component(label, point) {}
};

std::vector<Component> connected_components(PyArrayObject *image, Connectivity connectivity = CONNECTIVITY_8);
std::vector<ComponentWithEdge> connected_components_with_edge(PyArrayObject *image, Connectivity connectivity = CONNECTIVITY_8, Connectivity edge_connectivity = CONNECTIVITY_4);
