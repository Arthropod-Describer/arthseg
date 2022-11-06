#include "connected_components.hpp"
#include "utils.hpp"

static void dfs(PyArrayObject *image, Matrix<char> &marker, Component &component, Connectivity connectivity);

std::vector<Component> connected_components(PyArrayObject *image, Connectivity connectivity)
{
    const auto rows = PyArray_DIM(image, 0);
    const auto cols = PyArray_DIM(image, 1);
    Matrix<char> marker(rows, cols);
    std::vector<Component> components;

    for (auto row = 0; row < rows; row++) {
        for (auto col = 0; col < cols; col++) {
            if (PyArray_At(image, row, col) != 0 && marker.at(row, col) == 0) {
                auto point = Point(row, col);
                components.emplace_back(PyArray_At(image, row, col), point);
                marker.at(point) = 1;
                dfs(image, marker, components.back(), connectivity);
            }
        }
    }

    return components;
}

static void dfs(PyArrayObject *image, Matrix<char> &marker, Component &component, Connectivity connectivity)
{
    for (size_t i = 0; i < component.size(); i++) {
        const auto &point = component.nodes[i];
        for (size_t j = 0; j < connectivity; j++) {
            auto row = point.row + drow[j];
            auto col = point.col + dcol[j];
            if (!is_outside(image, row, col) && marker.at(row, col) == 0 && PyArray_At(image, row, col) == component.label) {
                marker.at(row, col) = 1;
                component.nodes.emplace_back(row, col);
            }
        }
    }
}
