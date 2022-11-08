#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/ndarraytypes.h>

#include "types.hpp"

std::vector<Point> skeletonization(PyArrayObject *image, const std::vector<Point> &points);
