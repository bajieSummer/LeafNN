
#include <Python.h>

//gcc -I/opt/local/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12  \
    -o matrix_multiply.so -shared matrix_multiply.c \
    -L/opt/local/Library/Frameworks/Python.framework/Versions/3.12/lib \
    -lpython3.12

// Function to multiply two matrices
// Function to multiply two matrices
// Function to multiply two matrices
typedef double DataType;
static PyObject* matrix_multiply(PyObject* self, PyObject* args) {
    PyObject *A_list, *B_list;
    int rows_A, cols_A, rows_B, cols_B;

    // Parse the input arguments (two lists of lists)
    if (!PyArg_ParseTuple(args, "OO", &A_list, &B_list)) {
        return NULL; // Error parsing arguments
    }

    // Get dimensions of matrix A
    rows_A = (int)PyList_Size(A_list);
    cols_A = (int)PyList_Size(PyList_GetItem(A_list, 0));

    // Get dimensions of matrix B
    rows_B = (int)PyList_Size(B_list);
    cols_B = (int)PyList_Size(PyList_GetItem(B_list, 0));

    // Check for compatible dimensions
    if (cols_A != rows_B) {
        PyErr_SetString(PyExc_ValueError, "Incompatible matrix dimensions.");
        return NULL;
    }

    // Create result matrix C
    PyObject *C_list = PyList_New(rows_A);
    for (int i = 0; i < rows_A; i++) {
        PyObject *row = PyList_New(cols_B);
        PyList_SetItem(C_list, i, row);
    }

    // Perform matrix multiplication
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            DataType sum = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                DataType a_val = (DataType)PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(A_list, i), k));
                DataType b_val = (DataType)PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(B_list, k), j));
                sum += a_val * b_val;
            }
            PyList_SetItem(PyList_GetItem(C_list, i), j, PyFloat_FromDouble(sum));
        }
    }

    return C_list; // Return the resulting matrix
}

// Method definitions
static PyMethodDef MatrixMethods[] = {
    {"multiply", matrix_multiply, METH_VARARGS, "Multiply two matrices."},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Module definition
static struct PyModuleDef matrixmodule = {
    PyModuleDef_HEAD_INIT,
    "matrix_multiply", // Name of the module
    NULL,              // Module documentation (may be NULL)
    -1,                // Size of per-interpreter state of the module
    MatrixMethods      // Method definitions
};

// Module initialization function
PyMODINIT_FUNC PyInit_matrix_multiply(void) {
    return PyModule_Create(&matrixmodule);
}