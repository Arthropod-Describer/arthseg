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
    auto *po = PyArray_GETPTR2(image, 0, 0);
    std::cout << "got pointer" << std::endl;
    auto *item = PyArray_GETITEM(image, (char *) po);
    std::cout << "got item" << std::endl;
    std::cout << PyLong_AsUnsignedLong(PyArray_GETITEM(image, (char *) PyArray_GETPTR2(image, 0, 0))) << std::endl;
    std::cout << "showed" << std::endl;
    std::cout << PyArray_At(image, 0, 0) << std::endl;

    std::cout << "Loops " << std::endl;

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

    std::cout << "return" << std::endl;

    return components;
}

static void dfs(PyArrayObject *image, Matrix<bool> &marker, Component &component, Connectivity connectivity)
{
    for (size_t i = 0; i < component.size(); i++) {
        const auto &point = component.nodes[i];
        for (size_t j = 0; j < connectivity; j++) {
            const auto row = point.row + drow[j];
            const auto col = point.col + dcol[j];
            if (!is_outside(image, row, col) && !marker.at(row, col) && PyArray_At(image, row, col) == component.label) {
                marker.at(row, col) = true;
                component.nodes.emplace_back(row, col);
            }
        }
    }
}
