#include "connected_components.hpp"
#include "utils.hpp"

#include <iostream>

static void dfs(PyArrayObject *image, Matrix<bool> &marker, Component &component, Connectivity connectivity);

std::vector<Component> connected_components(PyArrayObject *image, Connectivity connectivity)
{
    const auto rows = PyArray_DIM(image, 0);
    const auto cols = PyArray_DIM(image, 1);
    Matrix<bool> marker(rows, cols);
    std::vector<Component> components;

    for (npy_intp row = 0; row < rows; row++) {
        for (npy_intp col = 0; col < cols; col++) {
            if (PyArray_At(image, row, col) != 0 && !marker.at(row, col)) {
                const auto point = Point(row, col);
                components.emplace_back(PyArray_At(image, row, col), std::vector<Point>({ point }));
                marker.at(point) = true;
                dfs(image, marker, components.back(), connectivity);
            }
        }
    }

    return components;
}

static void dfs(PyArrayObject *image, Matrix<bool> &marker, Component &component, Connectivity connectivity)
{
    for (size_t i = 0; i < component.size(); i++) {
        const auto &point = component.nodes[i];
        for (size_t j = 0; j < connectivity; j++) {
            const auto row = point.row + drow[j];
            const auto col = point.col + dcol[j];
            bool debug = component.size() == 92170;
            if (debug)
                std::cout << j << " " << row << " " << col << " " << component.size() << std::endl;
            if (!is_outside(image, row, col, debug) && !marker.at(row, col) && PyArray_At(image, row, col, debug) == component.label) {
                if (debug)
                    std::cout << "adding point" << std::endl;
                marker.at(row, col) = true;
                component.nodes.emplace_back(row, col);
            }
        }
    }
}
