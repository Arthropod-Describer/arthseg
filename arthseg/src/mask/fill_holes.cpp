#include <numeric>

#include "connected_components.hpp"
#include "fill_holes.hpp"
#include "utils.hpp"

#include <iostream>

PyArrayObject *fill_holes(PyArrayObject *image, float hole_area)
{
    std::cout << "called fill holes" << std::endl;
    import_array();
    std::cout << "imported array" << std::endl;
    PyArrayObject *mask = (PyArrayObject *) PyArray_EMPTY(PyArray_NDIM(image), PyArray_DIMS(image), NPY_UINT8, 0);
    PyArrayObject *output = (PyArrayObject *) PyArray_Empty(PyArray_NDIM(image), PyArray_DIMS(image), PyArray_DTYPE(image), 0);
    if (mask == NULL || output == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }

    for (npy_intp row = 0; row < PyArray_DIM(image, 0); row++) {
        for (npy_intp col = 0; col < PyArray_DIM(image, 1); col++) {
            auto value = PyLong_AsUnsignedLong(PyArray_GETITEM(image, (char *) PyArray_GETPTR2(image, row, col)));
            PyArray_SETITEM(mask, (char *) PyArray_GETPTR2(mask, row, col), Py_BuildValue("B", value == 0));
        }
    }

    std::cout << "copied" << std::endl;

    auto components = connected_components(mask, CONNECTIVITY_4);

    std::cout << "connected components" << std::endl;

    auto area = std::accumulate(components.begin(), components.end(), 0, [](auto acc, auto &component) {
        return acc + component.size();
    });

    std::cout << "area" << std::endl;

    if (PyArray_CopyInto(output, image)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to copy image");
        return NULL;
    }

    std::cout << "copying" << std::endl;

    size_t max_area = hole_area * (PyArray_DIM(image, 0) * PyArray_DIM(image, 1) - area);

    for (auto &component : components) {
        if (component.size() < max_area) {
            for (auto &node : component.nodes) {
                PyArray_Set(output, node.row, node.col, 1);
            }
        }
    }

    std::cout << "output" << std::endl;
    return output;
}
