#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/arrayobject.h>

inline constexpr int drow[] = { 1, 0, -1, 0, 1, 1, -1, -1 };
inline constexpr int dcol[] = { 0, 1, 0, -1, 1, -1, 1, -1 };

inline bool is_outside(PyArrayObject *image, size_t row, size_t col)
{
    // _import_array();
    return (npy_intp) row >= PyArray_DIM(image, 0) || (npy_intp) col >= PyArray_DIM(image, 1);
}

inline unsigned long PyArray_At(PyArrayObject *image, size_t row, size_t col)
{
    // _import_array();
    return PyLong_AsUnsignedLong(PyArray_GETITEM(image, (char *) PyArray_GETPTR2(image, row, col)));
}

inline void PyArray_Set(PyArrayObject *image, size_t row, size_t col, unsigned long value)
{
    // _import_array();
    PyArray_SETITEM(image, (char *) PyArray_GETPTR2(image, row, col), Py_BuildValue("B", value));
}
