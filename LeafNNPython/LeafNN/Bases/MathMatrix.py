# config.py
import numpy as np
#import matrix_multiply
class MathMatrix:
    newaxis = np.newaxis
    default_Type = np.float64
    def array(*args, **kwargs):
        return np.array(*args, dtype=MathMatrix.default_Type, **kwargs)
    
    def ones(shape, **kwargs):
        """Create an array of ones with dtype float128."""
        return np.ones(shape, dtype=MathMatrix.default_Type, **kwargs)
    
    # Custom wrapper for np.zeros
    def zeros(shape, **kwargs):
        """Create an array of zeros with dtype float128."""
        return np.zeros(shape, dtype=MathMatrix.default_Type, **kwargs)

    # Custom wrapper for np.identity
    def identity(n):
        """Create an array of zeros with dtype float128."""
        return np.identity(n, dtype=MathMatrix.default_Type)

    def diag(v, k=0):
        """Create diag matrix with existed array v."""
        return np.diag(v,k)
    
    def rand(*args):
        """Generate random numbers and return them as float128."""
        return np.random.rand(*args).astype(MathMatrix.default_Type)
    
    def sum(a, axis=None, dtype=None, keepdims=False):
        """Calculate the sum and return as float128."""
        #return np.sum(a, axis=axis, dtype=dtype, keepdims=keepdims).astype(MathMatrix.default_Type)
        return np.sum(a, axis=axis, dtype=dtype, keepdims=keepdims)
    def sqrt(arr):
        """Compute the square root and return as float128."""
        #return np.sqrt(arr).astype(MathMatrix.default_Type)
        return np.sqrt(arr)
    
    def square(a):
        return np.square(a)
    
    def log(arr):
        """Compute the natural logarithm and return as float128 if necessary."""
        # Check if the input is already float128
        if arr.dtype == MathMatrix.default_Type:
            return np.log(arr)  # No need to convert
        else:
            return np.log(arr).astype(MathMatrix.default_Type)  # Convert to float128 if not
        
    def exp(arr):
        """Compute the exponential and return as float128 if necessary."""
        # Check if the input is already float128
        if arr.dtype == MathMatrix.default_Type:
            return np.exp(arr)  # No need to convert
        else:
            return np.exp(arr).astype(MathMatrix.default_Type)  # Convert to float128 if not
    
    def dot(a, b):
        """Compute the dot product of two arrays and return as float128."""
        return np.dot(a, b)
        #return result.astype(MathMatrix.default_Type)  # Convert to float128

    def matmul(a, b):
        """Perform matrix multiplication and return the result as float128."""
        return np.matmul(a, b)
        #return MathMatrix.matmulS(a,b)
    
    def matmulS(a,b):
        #result = matrix_multiply.multiply(a.tolist(),b.tolist())
        #return np.array(result)
        # a1 = np.ones([3,2])*2.0
        # a2 = np.ones([2,4])*1.0
        # res = matrix_multiply.multiply(a1.tolist(),a2.tolist())
        return np.matmul(a,b)
        # (m,n) = a.shape
        # (n1,mt) = b.shape
        # res = MathMatrix.zeros([m,mt])
        # for i in range(m):
        #     for j in range(mt):
        #         res[i][j] = 0
        #         for t in range(n):
        #             res[i][j]+= a[i][t]*b[t][j]
        # return res

    def abs(arr):
        """Compute the absolute value and return as float128."""
        return np.abs(arr)
        #return result.astype(MathMatrix.default_Type)  # Convert to float128
    
    def hstack(tup):
        """Horizontally stack arrays and return as float128."""
        return np.hstack(tup)
        #return result.astype(MathMatrix.default_Type)  # Convert to float128

    def vstack(tup):
        return np.vstack(tup)
        
    def transpose(arr, axes=None):
        """Transpose an array and return it."""
        return np.transpose(arr, axes=axes)
    
    def argmax(array,aixs):
        return np.argmax(array, axis=aixs)

    def finfo(dtype=None):
        """Return floating-point information for the specified dtype."""
        if dtype is None:
            return np.finfo(dtype=MathMatrix.default_Type)
        else:
            return np.finfo(dtype)
    
    def isreal(arr):
        """Check if elements are real and return a boolean array."""
        return np.isreal(arr)

    def isnan(arr):
        """Check for NaNs in the array and return a boolean array."""
        return np.isnan(arr)
    
    def isinf(arr):
        """Check for infinite values in the array and return a boolean array."""
        return np.isinf(arr)
    
    def isNum(other):
        if isinstance(other, (int, float)) or isinstance(other, np.float128) or isinstance(other,np.float64):
            return True
        else:
            return False
    
    def inf():
        return np.inf
    
    def isClose(a,b):
        return np.isclose(a,b)
    
    def isAllCloseZero(a):
        return np.isclose(a.any(),0.0)
    
    def arange(range):
        return np.arange(range)
    
    def randIndices(arraySize,chooseSize):
        return np.random.choice(arraySize, size=chooseSize, replace=False)

    def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis=0, *, device=None):
        return np.linspace(start=start,stop=stop,num=num,endpoint=endpoint,retstep=retstep,dtype=dtype,axis=axis,device=device)

    def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
        return np.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)

    def is_numpy_array(variable):
        return isinstance(variable, np.ndarray)

    def set_printoptions(precision,suppress,threshold=None):
        np.set_printoptions(precision=precision, suppress=suppress, threshold=threshold)

    def getSign(v):
        return np.sign(v)

    

        
# # Store the original np.array function
# original_np_array = np.array

# # Redefine np.array to use float128 by default
# def custom_np_array(*args, **kwargs):
#     # Set dtype to float128 if not specified
#     if 'dtype' not in kwargs:
#         kwargs['dtype'] = np.float128
#     return original_np_array(*args, **kwargs)

# # Replace np.array with the custom function
# np.array = custom_np_array
