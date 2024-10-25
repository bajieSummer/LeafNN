'''
Author: Sophie
email: bajie615@126.com
Date: 2024-05-31 18:16:27
Description: file content
'''
import numpy as np

# Create a 1D NumPy array
arr1d = np.array([1, 2, 3, 4, 5])
print("1D Array:")
print(arr1d)

# Create a 2D NumPy array
arr2d = np.array([[1, 3, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D Array:")
print(arr2d)

# Accessing array elements
print("\nAccessing Array Elements:")
print(arr2d[0, 1])  # Accessing element at row 0, column 1
print(arr2d[2])     # Accessing entire row at index 2

# Array shape and size
print("\nArray Shape and Size:")
print(arr2d.shape)  # Shape of the array (rows, columns)
print(arr2d.size)   # Total number of elements in the array

# Array operations
print("\nArray Operations:")
print(np.sum(arr1d))             # Sum of array elements
print(np.mean(arr2d))            # Mean of array elements
print(np.max(arr1d))             # Maximum value in the array
print(np.std(arr2d))             # Standard deviation of array elements
print(np.transpose(arr2d))       # Transpose of the array
print(np.linalg.inv(arr2d))      # Matrix inverse

# Array manipulation
print("\nArray Manipulation:")
reshaped_arr = arr1d.reshape((5, 1))  # Reshape the array to a 5x1 matrix
print(reshaped_arr)

# Array concatenation
arr3 = np.array([10, 11, 12])
arr_concatenated = np.concatenate((arr1d, arr3))
print(arr_concatenated)

# Array slicing
print("\nArray Slicing:")
print(arr1d[2:4])       # Slice elements from index 2 to 4 (exclusive)
print(arr2d[:, 1:])    # Slice all rows and columns starting from column 1

# Broadcasting
print("\nBroadcasting:")
arr_broadcasted = arr1d + 2
print(arr_broadcasted)

