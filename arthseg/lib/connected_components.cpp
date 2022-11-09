#include "connected_components.hpp"
#include "utils.hpp"

#include <iostream>

static void dfs(PyArrayObject *image, Matrix<bool> &marker, Component &component, Connectivity connectivity);

std::vector<Component> connected_components(PyArrayObject *image, Connectivity connectivity)
{
    std::cout << "connected_components" << std::endl;
    const auto rows = PyArray_DIM(image, 0);
    std::cout << "got rows" << std::endl;
    const auto cols = PyArray_DIM(image, 1);
    std::cout << "got cols" << std::endl;
    Matrix<bool> marker(rows, cols);
    std::vector<Component> components;

    std::cout << "initialized" << std::endl;

    for (npy_intp row = 0; row < rows; row++) {
        for (npy_intp col = 0; col < cols; col++) {
            if (PyArray_At(image, row, col) != 0 && !marker.at(row, col)) {
                auto point = Point(row, col);
                components.emplace_back(PyArray_At(image, row, col), point);
                marker.at(point) = true;
                dfs(image, marker, components.back(), connectivity);
            }
        }
    }

    std::cout << "return" << std::endl;

    return components;
}

static void dfs(PyArrayObject *image, Matrix<bool> &marker, Component &component, Connectivity connectivity)
{
    for (size_t i = 0; i < component.size(); i++) {
        const auto &point = component.nodes[i];
        for (size_t j = 0; j < connectivity; j++) {
            auto row = point.row + drow[j];
            auto col = point.col + dcol[j];
            if (!is_outside(image, row, col) && !marker.at(row, col) && PyArray_At(image, row, col) == component.label) {
                marker.at(row, col) = true;
                component.nodes.emplace_back(row, col);
            }
        }
    }
}
