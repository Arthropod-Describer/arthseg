#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/ndarraytypes.h>

PyArrayObject *fill_holes(PyArrayObject *image, float hole_area);
