#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "src/legs/legs.hpp"
#include "src/mask/fill_holes.hpp"
#include "src/mask/remove_dirt.hpp"
#include "src/regions/refine.hpp"

#include "skeletonization.hpp"

static PyObject *Py_RemoveDirt(PyObject *, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *image;
    int keep = true;
    size_t max_distance = 20;
    float min_area = 0.05;
    const char *kwlist[] = { "", "keep", "max_distance", "min_area", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|pIf", const_cast<char **>(kwlist), &PyArray_Type, &image, &keep, &max_distance, &min_area)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argumnets");
        return NULL;
    }

    return Py_BuildValue("O", remove_dirt(image, keep, max_distance, min_area));
}

static PyObject *Py_FillHoles(PyObject *, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *image;
    float hole_area = 0.001;
    const char *kwlist[] = { "", "hole_area", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|f", const_cast<char **>(kwlist), &PyArray_Type, &image, &hole_area)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argumnets");
        return NULL;
    }

    return Py_BuildValue("O", fill_holes(image, hole_area));
}

static PyObject *Py_RefineRegions(PyObject *, PyObject *args)
{
    PyArrayObject *image;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &image)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argumnets");
        return NULL;
    }

    return Py_BuildValue("O", refine_regions(image));
}

static PyObject *Py_RefineLegs(PyObject *, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *image;
    PyObject *pair_labels;
    PyObject *body_labels;
    const char *kwlist[] = { "", "pair_labels", "body_labels", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!", const_cast<char **>(kwlist), &PyArray_Type, &image, &PyList_Type, &pair_labels, &PySet_Type, &body_labels)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argumnets");
        return NULL;
    }

    std::vector<std::vector<Point>> legs;
    std::vector<Point> body;
    for (auto &component : connected_components(image)) {
        if (component.label == 4) {
            for (auto &leg : split_leg(image, body_labels, component)) {
                if (!leg.empty()) {
                    legs.push_back(std::move(leg));
                }
            }
        } else if (PySet_Contains(body_labels, PyLong_FromLong(component.label))) {
            body.insert(body.end(), component.nodes.begin(), component.nodes.end());
        }
    }

    reored_legs(image, body_labels, pair_labels, legs, body);

    // PyArrayObject *mask = (PyArrayObject *) PyArray_ZEROS(PyArray_NDIM(image), PyArray_DIMS(image), NPY_UINT8, 0);
    // std::vector<std::vector<Point>> legs;
    // for (auto &component : connected_components(image)) {
    //     if (component.label == 4) {
    //         for (auto &point : skeletonization(image, component.nodes)) {
    //             PyArray_SETITEM(mask, (char *) PyArray_GETPTR2(mask, point.row, point.col), PyLong_FromLong(1));
    //         }
    //     }
    // }

    return Py_BuildValue("O", image);
}

static PyObject *Py_LegSegments(PyObject *, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *image;
    PyObject *labels_map;
    PyObject *body_labels;
    const char *kwlist[] = { "", "labels", "body_labels", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!", const_cast<char **>(kwlist), &PyArray_Type, &image, &PyDict_Type, &labels_map, &PySet_Type, &body_labels)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argumnets");
        return NULL;
    }

    PyArrayObject *output = (PyArrayObject *) PyArray_Empty(PyArray_NDIM(image), PyArray_DIMS(image), PyArray_DTYPE(image), 0);
    if (output == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    if (PyArray_CopyInto(output, image)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to copy image");
        return NULL;
    }

    for (auto &component : connected_components(image)) {
        auto *labels = PyDict_GetItem(labels_map, PyLong_FromLong(component.label));
        if (labels != NULL) {
            leg_segments(output, labels, body_labels, component);
        }
    }

    return Py_BuildValue("O", output);
}

static PyMethodDef methods[] = {
    { "remove_dirt", (PyCFunction) Py_RemoveDirt, METH_VARARGS | METH_KEYWORDS, "" },
    { "fill_holes", (PyCFunction) Py_FillHoles, METH_VARARGS | METH_KEYWORDS, "" },
    { "refine_regions", (PyCFunction) Py_RefineRegions, METH_VARARGS, "" },
    { "refine_legs", (PyCFunction) Py_RefineLegs, METH_VARARGS | METH_KEYWORDS, "" },
    { "leg_segments", (PyCFunction) Py_LegSegments, METH_VARARGS | METH_KEYWORDS, "" },
    { NULL, NULL, 0, NULL },
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "arthseg",
    "Python C++ extensions for native proccesing.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_arthseg()
{
    import_array();
    return PyModule_Create(&module);
};
