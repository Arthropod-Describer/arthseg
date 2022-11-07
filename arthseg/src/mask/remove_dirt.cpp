#include <limits>

#include "connected_components.hpp"
#include "remove_dirt.hpp"
#include "utils.hpp"

static size_t min_distance(const std::vector<Point> &left, const std::vector<Point> &right);

PyArrayObject *remove_dirt(PyArrayObject *image, bool keep, size_t max_distance, float min_area)
{
    import_array();
    PyArrayObject *mask = (PyArrayObject *) PyArray_EMPTY(PyArray_NDIM(image), PyArray_DIMS(image), NPY_UINT8, 0);
    for (npy_intp row = 0; row < PyArray_DIM(image, 0); row++) {
        for (npy_intp col = 0; col < PyArray_DIM(image, 1); col++) {
            PyArray_SETITEM(mask, (char *) PyArray_GETPTR2(mask, row, col), Py_BuildValue("B", PyLong_AsUnsignedLong(PyArray_GETITEM(image, (char *) PyArray_GETPTR2(image, row, col))) != 0));
        }
    }

    auto components = connected_components_with_edge(mask);
    if (components.size() < 2) {
        return image;
    }

    auto largest = std::max_element(components.begin(), components.end(), [](auto &left, auto &right) {
        return left.size() < right.size();
    });

    for (auto it = components.begin(); it != components.end(); it++) {
        if (it == largest) {
            continue;
        }
        if (!keep || it->size() < min_area * largest->size() || min_distance(largest->edge, it->edge) > max_distance) {
            for (auto &node : it->nodes) {
                PyArray_Set(image, node.row, node.col, 0);
            }
        }
    }

    return image;
}

static size_t min_distance(const std::vector<Point> &left, const std::vector<Point> &right)
{
    size_t distance = std::numeric_limits<size_t>::max();

    for (auto &point1 : left) {
        for (auto &point2 : right) {
            distance = std::min(distance, (size_t) Point::distance(point1, point2));
        }
    }
    return distance;
}
