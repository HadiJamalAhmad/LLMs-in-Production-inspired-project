# fibonacci.py  # This file contains the Fibonacci function being tested.

def fibonacci_sequence(n):  # Define a function named fibonacci_sequence that takes n as input.
    """Returns the nth number in the Fibonacci sequence"""  # Describe what the function returns.

    if n < 1:  # Check whether n is invalid because this sequence starts at position 1.
        raise ValueError("n must be a positive integer")  # Raise ValueError for invalid input.

    if n == 1:  # Check whether the requested position is the first Fibonacci number.
        return 0  # Return 0 because the first Fibonacci number in this version is 0.

    if n == 2:  # Check whether the requested position is the second Fibonacci number.
        return 1  # Return 1 because the second Fibonacci number is 1.

    previous = 0  # Store the Fibonacci number at position 1.
    current = 1  # Store the Fibonacci number at position 2.

    for _ in range(3, n + 1):  # Loop from the third position up to n.
        previous, current = current, previous + current  # Move forward one Fibonacci step.

    return current  # Return the nth Fibonacci number.