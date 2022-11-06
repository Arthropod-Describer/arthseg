#include <map>

#include "connected_components.hpp"
#include "refine.hpp"
#include "utils.hpp"

#include <iostream>

static void attach(PyArrayObject *image, const ComponentWithEdge &component);

PyArrayObject *refine_regions(PyArrayObject *image)
{
    auto components = connected_components_with_edge(image);
    std::sort(components.begin(), components.end());

    size_t label = 1;
    size_t body_area = 0;
    for (auto &component : components) {
        std::cout << component.label << " " << component.size() << std::endl;
        if (component.label == label || (component.label == 4 && component.size() > body_area / 40)) {
            label++;
            if (component.label < 4) {
                body_area += component.size();
            }
        } else {
            attach(image, component);
        }
    }

    return image;
}

static void attach(PyArrayObject *image, const ComponentWithEdge &component)
{
    std::map<unsigned long, size_t> neighbours;

    for (auto &edge : component.edge) {
        for (size_t i = 0; i < 4; i++) {
            auto row = edge.row + drow[i];
            auto col = edge.col + dcol[i];

            if (!is_outside(image, row, col) && PyArray_At(image, row, col) != 0 && PyArray_At(image, row, col) != component.label) {
                auto label = PyArray_At(image, row, col);
                neighbours.try_emplace(label, 0);
                neighbours[label]++;
            }
        }
    }

    if (neighbours.empty()) {
        return;
    }

    auto [label, _] = *std::max_element(neighbours.begin(), neighbours.end(), [](const auto &l, const auto &r) { return l.second < r.second; });
    for (auto &node : component.nodes) {
        PyArray_Set(image, node.row, node.col, label);
    }
}
