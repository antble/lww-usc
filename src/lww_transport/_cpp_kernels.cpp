#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <sstream>

namespace {

constexpr int INPUT_ARRAY_FLAGS = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
constexpr int OUTPUT_ARRAY_FLAGS = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_WRITEABLE;

PyArrayObject* require_double_array(PyObject* obj, int flags, const char* name) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(obj, NPY_DOUBLE, flags)
    );
    if (arr == nullptr) {
        std::ostringstream message;
        message << name << " must be a float64 NumPy array";
        PyErr_SetString(PyExc_TypeError, message.str().c_str());
        return nullptr;
    }
    return arr;
}

inline double get1(const double* data, npy_intp s0, int i) {
    return *reinterpret_cast<const double*>(
        reinterpret_cast<const char*>(data) + static_cast<npy_intp>(i) * s0
    );
}

inline double get2(const double* data, npy_intp s0, npy_intp s1, int i, int j) {
    return *reinterpret_cast<const double*>(
        reinterpret_cast<const char*>(data) + static_cast<npy_intp>(i) * s0 + static_cast<npy_intp>(j) * s1
    );
}

inline double& ref1(double* data, npy_intp s0, int i) {
    return *reinterpret_cast<double*>(
        reinterpret_cast<char*>(data) + static_cast<npy_intp>(i) * s0
    );
}

inline double& ref2(double* data, npy_intp s0, npy_intp s1, int i, int j) {
    return *reinterpret_cast<double*>(
        reinterpret_cast<char*>(data) + static_cast<npy_intp>(i) * s0 + static_cast<npy_intp>(j) * s1
    );
}

bool check_1d(PyArrayObject* arr, npy_intp expected, const char* name) {
    if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) != expected) {
        std::ostringstream message;
        message << name << " must have shape (" << expected << ",)";
        PyErr_SetString(PyExc_ValueError, message.str().c_str());
        return false;
    }
    return true;
}

bool check_2d(PyArrayObject* arr, npy_intp dim0, npy_intp dim1, const char* name) {
    if (PyArray_NDIM(arr) != 2 || PyArray_DIM(arr, 0) != dim0 || PyArray_DIM(arr, 1) != dim1) {
        std::ostringstream message;
        message << name << " must have shape (" << dim0 << ", " << dim1 << ")";
        PyErr_SetString(PyExc_ValueError, message.str().c_str());
        return false;
    }
    return true;
}

PyObject* fill_potential_banded(PyObject*, PyObject* args) {
    PyObject* amx_obj = nullptr;
    PyObject* rvs_obj = nullptr;
    PyObject* sin_obj = nullptr;
    int n = 0;
    int nx = 0;
    if (!PyArg_ParseTuple(args, "OOOii", &amx_obj, &rvs_obj, &sin_obj, &n, &nx)) {
        return nullptr;
    }

    PyArrayObject* amx_arr = require_double_array(amx_obj, OUTPUT_ARRAY_FLAGS, "amx");
    if (amx_arr == nullptr) return nullptr;
    PyArrayObject* rvs_arr = require_double_array(rvs_obj, INPUT_ARRAY_FLAGS, "rvs");
    if (rvs_arr == nullptr) {
        Py_DECREF(amx_arr);
        return nullptr;
    }
    PyArrayObject* sin_arr = require_double_array(sin_obj, INPUT_ARRAY_FLAGS, "sin_matrix");
    if (sin_arr == nullptr) {
        Py_DECREF(amx_arr);
        Py_DECREF(rvs_arr);
        return nullptr;
    }

    const int rows = 4 * n + 1;
    const int cols = n * nx;
    const int half = n / 2;
    if (!check_2d(amx_arr, rows, cols, "amx") ||
        !check_1d(rvs_arr, nx, "rvs") ||
        !check_2d(sin_arr, n - 1, half, "sin_matrix")) {
        PyArray_DiscardWritebackIfCopy(amx_arr);
        Py_DECREF(amx_arr);
        Py_DECREF(rvs_arr);
        Py_DECREF(sin_arr);
        return nullptr;
    }

    double* amx = static_cast<double*>(PyArray_DATA(amx_arr));
    const double* rvs = static_cast<const double*>(PyArray_DATA(rvs_arr));
    const double* sin_matrix = static_cast<const double*>(PyArray_DATA(sin_arr));
    const npy_intp amx_s0 = PyArray_STRIDE(amx_arr, 0);
    const npy_intp amx_s1 = PyArray_STRIDE(amx_arr, 1);
    const npy_intp rvs_s0 = PyArray_STRIDE(rvs_arr, 0);
    const npy_intp sin_s0 = PyArray_STRIDE(sin_arr, 0);
    const npy_intp sin_s1 = PyArray_STRIDE(sin_arr, 1);
    const int center = 2 * n;
    const double scale = 2.0 / static_cast<double>(n);

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static) if(nx > 8)
    for (int ix = 0; ix < nx; ++ix) {
        const int base = ix * n;
        for (int j0 = 0; j0 < n - 1; ++j0) {
            double val = 0.0;
            for (int ip0 = 0; ip0 < half; ++ip0) {
                const int ip = ip0 + 1;
                int right = ix + ip;
                if (right >= nx) right = nx - 1;
                int left = ix - ip;
                if (left < 0) left = 0;
                val += get2(sin_matrix, sin_s0, sin_s1, j0, ip0) *
                       (get1(rvs, rvs_s0, right) - get1(rvs, rvs_s0, left));
            }
            val *= scale;
            if (val == 0.0) continue;

            const int width = n - (j0 + 1);
            const int upper_row = center + (j0 + 1);
            const int lower_row = center - (j0 + 1);
            for (int offset = 0; offset < width; ++offset) {
                ref2(amx, amx_s0, amx_s1, upper_row, base + offset) = -val;
                ref2(amx, amx_s0, amx_s1, lower_row, base + (j0 + 1) + offset) = val;
            }
        }
    }
    Py_END_ALLOW_THREADS

    PyArray_ResolveWritebackIfCopy(amx_arr);
    Py_DECREF(amx_arr);
    Py_DECREF(rvs_arr);
    Py_DECREF(sin_arr);
    Py_RETURN_NONE;
}

PyObject* fill_scattering_banded(PyObject*, PyObject* args) {
    PyObject* S_obj = nullptr;
    PyObject* b_obj = nullptr;
    int n = 0;
    int nx = 0;
    double tcol = 0.0;
    if (!PyArg_ParseTuple(args, "OOiid", &S_obj, &b_obj, &n, &nx, &tcol)) {
        return nullptr;
    }

    PyArrayObject* S_arr = require_double_array(S_obj, OUTPUT_ARRAY_FLAGS, "S");
    if (S_arr == nullptr) return nullptr;
    PyArrayObject* b_arr = require_double_array(b_obj, INPUT_ARRAY_FLAGS, "b");
    if (b_arr == nullptr) {
        PyArray_DiscardWritebackIfCopy(S_arr);
        Py_DECREF(S_arr);
        return nullptr;
    }

    const int rows = 4 * n + 1;
    const int cols = n * nx;
    if (!check_2d(S_arr, rows, cols, "S") || !check_1d(b_arr, cols, "b")) {
        PyArray_DiscardWritebackIfCopy(S_arr);
        Py_DECREF(S_arr);
        Py_DECREF(b_arr);
        return nullptr;
    }

    double* S = static_cast<double*>(PyArray_DATA(S_arr));
    const double* b = static_cast<const double*>(PyArray_DATA(b_arr));
    const npy_intp S_s0 = PyArray_STRIDE(S_arr, 0);
    const npy_intp S_s1 = PyArray_STRIDE(S_arr, 1);
    const npy_intp b_s0 = PyArray_STRIDE(b_arr, 0);
    const int center = 2 * n;

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static) if(nx > 4)
    for (int ix = 0; ix < nx; ++ix) {
        const int base = ix * n;
        double rho = 0.0;
        for (int j = 0; j < n; ++j) {
            rho += get1(b, b_s0, base + j);
        }

        for (int j = 0; j < n; ++j) {
            double weight = 0.0;
            if (rho != 0.0) {
                weight = tcol * get1(b, b_s0, base + j) / rho;
            }
            for (int jp = 0; jp < n; ++jp) {
                ref2(S, S_s0, S_s1, center + j - jp, base + jp) += weight;
            }
            ref2(S, S_s0, S_s1, center, base + j) -= tcol;
        }
    }
    Py_END_ALLOW_THREADS

    PyArray_ResolveWritebackIfCopy(S_arr);
    Py_DECREF(S_arr);
    Py_DECREF(b_arr);
    Py_RETURN_NONE;
}

PyObject* curcalc_density(PyObject*, PyObject* args) {
    PyObject* B_obj = nullptr;
    PyObject* rj_obj = nullptr;
    int nx = 0;
    int n = 0;
    double coef = 0.0;
    if (!PyArg_ParseTuple(args, "OOiid", &B_obj, &rj_obj, &nx, &n, &coef)) {
        return nullptr;
    }

    PyArrayObject* B_arr = require_double_array(B_obj, INPUT_ARRAY_FLAGS, "B");
    if (B_arr == nullptr) return nullptr;
    PyArrayObject* rj_arr = require_double_array(rj_obj, OUTPUT_ARRAY_FLAGS, "rj");
    if (rj_arr == nullptr) {
        Py_DECREF(B_arr);
        return nullptr;
    }

    if (!check_2d(B_arr, nx, n, "B") || !check_1d(rj_arr, nx, "rj")) {
        Py_DECREF(B_arr);
        PyArray_DiscardWritebackIfCopy(rj_arr);
        Py_DECREF(rj_arr);
        return nullptr;
    }

    const double* B = static_cast<const double*>(PyArray_DATA(B_arr));
    double* rj = static_cast<double*>(PyArray_DATA(rj_arr));
    const npy_intp B_s0 = PyArray_STRIDE(B_arr, 0);
    const npy_intp B_s1 = PyArray_STRIDE(B_arr, 1);
    const npy_intp rj_s0 = PyArray_STRIDE(rj_arr, 0);

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static) if(nx > 16)
    for (int ix = 0; ix < nx; ++ix) {
        double total = 0.0;
        for (int j = 0; j < n; ++j) {
            total += get2(B, B_s0, B_s1, ix, j);
        }
        ref1(rj, rj_s0, ix) = coef * total;
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(B_arr);
    PyArray_ResolveWritebackIfCopy(rj_arr);
    Py_DECREF(rj_arr);
    Py_RETURN_NONE;
}

PyObject* curcalc_current(PyObject*, PyObject* args) {
    PyObject* B_obj = nullptr;
    PyObject* rj_obj = nullptr;
    int nx = 0;
    int n = 0;
    double cofj = 0.0;
    if (!PyArg_ParseTuple(args, "OOiid", &B_obj, &rj_obj, &nx, &n, &cofj)) {
        return nullptr;
    }

    PyArrayObject* B_arr = require_double_array(B_obj, INPUT_ARRAY_FLAGS, "B");
    if (B_arr == nullptr) return nullptr;
    PyArrayObject* rj_arr = require_double_array(rj_obj, OUTPUT_ARRAY_FLAGS, "rj");
    if (rj_arr == nullptr) {
        Py_DECREF(B_arr);
        return nullptr;
    }

    if (!check_2d(B_arr, nx, n, "B") || !check_1d(rj_arr, nx, "rj")) {
        Py_DECREF(B_arr);
        PyArray_DiscardWritebackIfCopy(rj_arr);
        Py_DECREF(rj_arr);
        return nullptr;
    }

    const double* B = static_cast<const double*>(PyArray_DATA(B_arr));
    double* rj = static_cast<double*>(PyArray_DATA(rj_arr));
    const npy_intp B_s0 = PyArray_STRIDE(B_arr, 0);
    const npy_intp B_s1 = PyArray_STRIDE(B_arr, 1);
    const npy_intp rj_s0 = PyArray_STRIDE(rj_arr, 0);
    const int nh = n / 2;

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static) if(nx > 16)
    for (int ix = 0; ix < nx; ++ix) {
        ref1(rj, rj_s0, ix) = 0.0;
    }

    #pragma omp parallel for schedule(static) if(nx > 16)
    for (int i0 = 1; i0 < nx - 2; ++i0) {
        double total = 0.0;
        for (int j = 0; j < nh; ++j) {
            const double weight = 2.0 * static_cast<double>(j + 1) - static_cast<double>(n) - 1.0;
            total += weight * (3.0 * get2(B, B_s0, B_s1, i0 + 1, j) -
                               get2(B, B_s0, B_s1, i0 + 2, j));
        }
        for (int j = nh; j < n; ++j) {
            const double weight = 2.0 * static_cast<double>(j + 1) - static_cast<double>(n) - 1.0;
            total += weight * (-get2(B, B_s0, B_s1, i0 - 1, j) +
                               3.0 * get2(B, B_s0, B_s1, i0, j));
        }
        ref1(rj, rj_s0, i0) = cofj * total;
    }
    Py_END_ALLOW_THREADS

    ref1(rj, rj_s0, 1) = 2.0 * get1(rj, rj_s0, 2) - get1(rj, rj_s0, 3);
    ref1(rj, rj_s0, 0) = get1(rj, rj_s0, 1);
    ref1(rj, rj_s0, nx - 2) = 2.0 * get1(rj, rj_s0, nx - 3) - get1(rj, rj_s0, nx - 4);
    ref1(rj, rj_s0, nx - 1) = get1(rj, rj_s0, nx - 2);

    Py_DECREF(B_arr);
    PyArray_ResolveWritebackIfCopy(rj_arr);
    Py_DECREF(rj_arr);
    Py_RETURN_NONE;
}

PyObject* openmp_enabled(PyObject*, PyObject*) {
#ifdef _OPENMP
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

PyObject* openmp_threads(PyObject*, PyObject*) {
#ifdef _OPENMP
    return PyLong_FromLong(omp_get_max_threads());
#else
    return PyLong_FromLong(1);
#endif
}

PyMethodDef methods[] = {
    {"fill_potential_banded", fill_potential_banded, METH_VARARGS, "Fill banded potential matrix in-place."},
    {"fill_scattering_banded", fill_scattering_banded, METH_VARARGS, "Fill banded scattering matrix in-place."},
    {"curcalc_density", curcalc_density, METH_VARARGS, "Compute density vector in-place."},
    {"curcalc_current", curcalc_current, METH_VARARGS, "Compute current vector in-place."},
    {"openmp_enabled", openmp_enabled, METH_NOARGS, "Return whether this extension was built with OpenMP."},
    {"openmp_threads", openmp_threads, METH_NOARGS, "Return OpenMP max threads, or 1 without OpenMP."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_cpp_kernels",
    "C++ kernels for lww_transport.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__cpp_kernels() {
    import_array();
    return PyModule_Create(&module);
}
