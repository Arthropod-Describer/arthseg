#include <map>

#include "connected_components.hpp"
#include "refine.hpp"
#include "utils.hpp"

static void attach(PyArrayObject *image, const ComponentWithEdge &component);

PyArrayObject *refine_regions(PyArrayObject *image)
{
    _import_array();
    PyArrayObject *output = (PyArrayObject *) PyArray_Empty(PyArray_NDIM(image), PyArray_DIMS(image), PyArray_DTYPE(image), 0);
    if (output == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    if (PyArray_CopyInto(output, image)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to copy image");
        return NULL;
    }

    auto components = connected_components_with_edge(image);

    std::map<size_t, const ComponentWithEdge *> max_components;
    size_t body_area = 0;
    for (const auto &component : components) {
        max_components.try_emplace(component.label, &component);
        if (max_components[component.label]->size() < component.size()) {
            max_components[component.label] = &component;
        }
        if (component.label < 4) {
            body_area += component.size();
        }
    }

    for (const auto &component : components) {
        if (&component == max_components[component.label] || (component.label == 4 && component.size() > body_area / 40)) {
            continue;
        }
        attach(output, component);
    }

    return output;
}

static void attach(PyArrayObject *image, const ComponentWithEdge &component)
{
    std::map<unsigned long, size_t> neighbours;

    for (auto &edge : component.edge) {
        for (size_t i = 0; i < CONNECTIVITY_4; i++) {
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

    auto [label, _] = *std::max_element(neighbours.begin(), neighbours.end(), [](auto &l, auto &r) { return l.second < r.second; });
    for (auto &node : component.nodes) {
        PyArray_Set(image, node.row, node.col, label);
    }
}
