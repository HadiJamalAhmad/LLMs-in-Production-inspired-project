''' fibonacci.py  # Start a multiline string showing the expected fibonacci.py file name.
def fibonacci_sequence(n):  # Define a function named fibonacci_sequence that takes n as input.
    """Returns the nth number in the Fibonacci sequence"""  # Document that the function returns the nth Fibonacci number.
'''  # End the multiline string.

import pytest  # Import pytest so the file can run tests and check expected exceptions.
import time  # Import time so the script can measure how long the tests take.
from fibonacci import fibonacci_sequence  # Import the fibonacci_sequence function from the fibonacci module.


def test_fibonacci_sequence():  # Define a pytest test function for fibonacci_sequence.
    test_cases = [(1, 0), (2, 1), (6, 5), (15, 377)]  # Store input n values and their expected Fibonacci outputs.

    for n, expected in test_cases:  # Loop through each test input and expected result.
        result = fibonacci_sequence(n)  # Call fibonacci_sequence with the current n value.
        assert (  # Start an assertion that checks whether the actual result matches the expected result.
            result == expected  # Compare the returned result to the expected Fibonacci value.
        ), f"Expected {expected}, but got {result} for n={n}."  # Show a helpful error message if the assertion fails.

    with pytest.raises(ValueError):  # Check that calling the function with invalid input raises a ValueError.
        fibonacci_sequence(-1)  # Call fibonacci_sequence with -1 to trigger the expected ValueError.


# Run tests using pytest and time it  # Explain that the following block runs pytest and measures execution time.
if __name__ == "__main__":  # Run this block only when the file is executed directly.
    start_time = time.time()  # Record the start time before running the tests.
    pytest.main(["-v", __file__])  # Run pytest in verbose mode on this exact file.
    end_time = time.time()  # Record the end time after pytest finishes.
    execution_time = end_time - start_time  # Calculate the elapsed test execution time in seconds.
    print(f"Execution time: {execution_time} seconds")  # Print the measured execution time.