#include "connected_components.hpp"
#include "utils.hpp"

static void dfs(PyArrayObject *image, Matrix<char> &marker, ComponentWithEdge &component, Connectivity connectivity, Connectivity edge_connectivity);
static bool is_edge(PyArrayObject *image, const Point &point, Connectivity edge_connectivity);

std::vector<ComponentWithEdge> connected_components_with_edge(PyArrayObject *image, Connectivity connectivity, Connectivity edge_connectivity)
{
    const auto rows = PyArray_DIM(image, 0);
    const auto cols = PyArray_DIM(image, 1);
    Matrix<char> marker(rows, cols);
    std::vector<ComponentWithEdge> components;

    for (auto row = 0; row < rows; row++) {
        for (auto col = 0; col < cols; col++) {
            if (PyArray_At(image, row, col) != 0 && marker.at(row, col) == 0) {
                auto point = Point(row, col);
                components.emplace_back(PyArray_At(image, row, col), point);
                components.back().edge.push_back(point);
                marker.at(point) = 1;
                dfs(image, marker, components.back(), connectivity, edge_connectivity);
            }
        }
    }

    return components;
}

static void dfs(PyArrayObject *image, Matrix<char> &marker, ComponentWithEdge &component, Connectivity connectivity, Connectivity edge_connectivity)
{
    for (size_t i = 0; i < component.size(); i++) {
        const auto &point = component.nodes[i];
        for (size_t j = 0; j < connectivity; j++) {
            auto row = point.row + drow[j];
            auto col = point.col + dcol[j];
            if (!is_outside(image, row, col) && marker.at(row, col) == 0 && PyArray_At(image, row, col) == component.label) {
                marker.at(row, col) = 1;
                component.nodes.emplace_back(row, col);

                if (is_edge(image, component.nodes.back(), edge_connectivity)) {
                    component.edge.push_back(component.nodes.back());
                }
            }
        }
    }
}

static bool is_edge(PyArrayObject *image, const Point &point, Connectivity edge_connectivity)
{
    for (size_t i = 0; i < edge_connectivity; i++) {
        auto row = point.row + drow[i];
        auto col = point.col + dcol[i];
        if (!is_outside(image, row, col) && PyArray_At(image, row, col) != PyArray_At(image, point.row, point.col)) {
            return true;
        }
    }
    return false;
}
