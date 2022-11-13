#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "src/legs/legs.hpp"
#include "src/mask/fill_holes.hpp"
#include "src/mask/remove_dirt.hpp"
#include "src/regions/refine.hpp"

static PyObject *Py_RemoveDirt(PyObject *, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *image = nullptr;
    int keep = true; // has to be type int
    size_t max_distance = 20;
    float min_area = 0.05;
    const char *kwlist[] = { "", "keep", "max_distance", "min_area", NULL };
    if (!PyArg_ParseTupleAndKeywords(args,
                kwargs,
                "O!|pIf",
                const_cast<char **>(kwlist),
                &PyArray_Type,
                &image,
                &keep,
                &max_distance,
                &min_area)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argumnets");
        return NULL;
    }

    return Py_BuildValue("O", remove_dirt(image, keep, max_distance, min_area));
}

static PyObject *Py_FillHoles(PyObject *, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *image = nullptr;
    float hole_area = 0.001;
    const char *kwlist[] = { "", "hole_area", NULL };
    if (!PyArg_ParseTupleAndKeywords(args,
                kwargs,
                "O!|f",
                const_cast<char **>(kwlist),
                &PyArray_Type,
                &image,
                &hole_area)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argumnets");
        return NULL;
    }

    return Py_BuildValue("O", fill_holes(image, hole_area));
}

static PyObject *Py_RefineRegions(PyObject *, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *image = nullptr;
    PyObject *body_labels = nullptr;
    float min_area = 0.01;
    const char *kwlist[] = { "", "body_labels", "min_area", NULL };
    if (!PyArg_ParseTupleAndKeywords(args,
                kwargs,
                "O!O!|f",
                const_cast<char **>(kwlist),
                &PyArray_Type,
                &image,
                &PySet_Type,
                &body_labels,
                &min_area)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argumnets");
        return NULL;
    }

    return Py_BuildValue("O", refine_regions(image, body_labels, min_area));
}

static PyObject *Py_RefineLegs(PyObject *, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *image = nullptr;
    PyObject *leg_labels = nullptr;
    PyObject *pair_labels = nullptr;
    PyObject *body_labels = nullptr;
    PyObject *alternative_labels = PySet_New(nullptr);
    const char *kwlist[] = { "",
        "leg_labels",
        "pair_labels",
        "body_labels",
        "alternative_labels",
        NULL };
    if (!PyArg_ParseTupleAndKeywords(args,
                kwargs,
                "O!O!O!O!|O!",
                const_cast<char **>(kwlist),
                &PyArray_Type,
                &image,
                &PySet_Type,
                &leg_labels,
                &PyList_Type,
                &pair_labels,
                &PySet_Type,
                &body_labels,
                &PySet_Type,
                &alternative_labels)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argumnets");
        return NULL;
    }

    PyArrayObject *output = (PyArrayObject *) PyArray_Empty(
            PyArray_NDIM(image), PyArray_DIMS(image), PyArray_DTYPE(image), 0);
    if (output == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    if (PyArray_CopyInto(output, image)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to copy image");
        return NULL;
    }

    std::vector<std::vector<Point>> legs;
    std::vector<Point> body;
    for (const auto &component : connected_components(output)) {
        auto *label = PyLong_FromLong(component.label);
        if (PySet_Contains(leg_labels, label)) {
            for (const auto &leg : split_leg(output, body_labels, alternative_labels, component)) {
                legs.emplace_back(std::move(leg));
            }
        } else if (PySet_Contains(body_labels, label) ||
                PySet_Contains(alternative_labels, label)) {
            body.insert(body.end(), component.nodes.begin(), component.nodes.end());
        }
    }

    reored_legs(
            output, body_labels, pair_labels, alternative_labels, legs, body);
    return Py_BuildValue("O", output);
}

static PyObject *Py_LegSegments(PyObject *, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *image = nullptr;
    PyObject *labels_map = nullptr;
    PyObject *body_labels = nullptr;
    PyObject *alternative_labels = PySet_New(nullptr);
    const char *kwlist[] = {
        "", "labels", "body_labels", "alternative_labels", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args,
                kwargs,
                "O!O!O!|O!",
                const_cast<char **>(kwlist),
                &PyArray_Type,
                &image,
                &PyDict_Type,
                &labels_map,
                &PySet_Type,
                &body_labels,
                &PySet_Type,
                &alternative_labels)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argumnets");
        return NULL;
    }

    PyArrayObject *output = (PyArrayObject *) PyArray_Empty(
            PyArray_NDIM(image), PyArray_DIMS(image), PyArray_DTYPE(image), 0);
    if (output == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
    if (PyArray_CopyInto(output, image)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to copy image");
        return NULL;
    }

    for (const auto &component : connected_components(output)) {
        auto *labels =
                PyDict_GetItem(labels_map, PyLong_FromLong(component.label));
        if (labels != NULL) {
            leg_segments(
                    output, labels, body_labels, alternative_labels, component);
        }
    }

    return Py_BuildValue("O", output);
}

static PyMethodDef methods[] = {
    { "remove_dirt", (PyCFunction) Py_RemoveDirt, METH_VARARGS | METH_KEYWORDS, "" },
    { "fill_holes", (PyCFunction) Py_FillHoles, METH_VARARGS | METH_KEYWORDS, "" },
    { "refine_regions", (PyCFunction) Py_RefineRegions, METH_VARARGS | METH_KEYWORDS, "" },
    { "refine_legs", (PyCFunction) Py_RefineLegs, METH_VARARGS | METH_KEYWORDS, "" },
    { "leg_segments", (PyCFunction) Py_LegSegments, METH_VARARGS | METH_KEYWORDS, "" },
    { NULL, NULL, 0, NULL },
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "arthseg",
    "Python C++ extensions for arthropod segmentation.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_arthseg()
{
    import_array();
    return PyModule_Create(&module);
};
