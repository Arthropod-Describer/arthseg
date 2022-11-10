#pragma once
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/arrayobject.h>

#include <iostream>

inline constexpr int drow[] = { 1, 0, -1, 0, 1, 1, -1, -1 };
inline constexpr int dcol[] = { 0, 1, 0, -1, 1, -1, 1, -1 };

inline bool is_outside(PyArrayObject *image, npy_intp row, npy_intp col)
{
    std::cout << row << " " << col << std::endl;
    bool is_out = row < 0 || col < 0 || row >= PyArray_DIM(image, 0) || col >= PyArray_DIM(image, 1);
    std::cout << is_out << std::endl;
    return is_out;
}

inline unsigned long PyArray_At(PyArrayObject *image, size_t row, size_t col)
{
    return PyLong_AsUnsignedLong(PyArray_GETITEM(image, (char *) PyArray_GETPTR2(image, row, col)));
}

inline void PyArray_Set(PyArrayObject *image, size_t row, size_t col, unsigned long value)
{
    PyArray_SETITEM(image, (char *) PyArray_GETPTR2(image, row, col), Py_BuildValue("B", value));
}
