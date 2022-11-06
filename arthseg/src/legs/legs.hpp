#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/ndarraytypes.h>

#include "connected_components.hpp"

std::vector<std::vector<Point>> split_leg(PyArrayObject *image, PyObject *body_labels, const Component &component);

void leg_segments(PyArrayObject *image, PyObject *labels, PyObject *body_labels, const Component &component);

void reored_legs(PyArrayObject *image, PyObject *labels, PyObject *pair_labels, PyObject *body_labels, const std::vector<std::vector<Point>> &legs);
