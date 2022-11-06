#include <numeric>

#include "connected_components.hpp"
#include "fill_holes.hpp"
#include "utils.hpp"

PyArrayObject *fill_holes(PyArrayObject *image, float hole_area)
{
    import_array();
    PyArrayObject *mask = (PyArrayObject *) PyArray_EMPTY(PyArray_NDIM(image), PyArray_DIMS(image), NPY_UINT8, 0);
    for (npy_intp row = 0; row < PyArray_DIM(image, 0); row++) {
        for (npy_intp col = 0; col < PyArray_DIM(image, 1); col++) {
            PyArray_SETITEM(mask, (char *) PyArray_GETPTR2(mask, row, col), Py_BuildValue("B", PyLong_AsUnsignedLong(PyArray_GETITEM(image, (char *) PyArray_GETPTR2(image, row, col))) == 0));
        }
    }

    auto components = connected_components(mask);
    auto area = std::accumulate(components.begin(), components.end(), 0, [](auto acc, auto &component) {
        return acc + component.size();
    });

    size_t max_area = hole_area * (PyArray_DIM(image, 0) * PyArray_DIM(image, 1) - area);
    for (auto &component : components) {
        if (component.size() < max_area) {
            for (auto &node : component.nodes) {
                PyArray_Set(image, node.row, node.col, 1);
            }
        }
    }

    return image;
}
