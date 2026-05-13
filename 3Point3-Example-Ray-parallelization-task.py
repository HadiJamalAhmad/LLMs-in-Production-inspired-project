import ray  # Import the Ray library, which enables parallel and distributed execution
import time  # Import the time module so we can pause execution using time.sleep()

ray.init()  # Start Ray and initialize the Ray runtime


# Define a regular Python function
def slow_function(x):  # Define a normal Python function that takes one argument, x
    time.sleep(1)  # Pause the function for 1 second to simulate slow work
    return x  # Return the input value unchanged


# Turn the function into a Ray task
@ray.remote  # Decorate the function so Ray can execute it remotely/asynchronously
def slow_function_ray(x):  # Define a Ray remote function that takes one argument, x
    time.sleep(1)  # Pause the function for 1 second to simulate slow work
    return x  # Return the input value unchanged


# Execute the slow function without Ray (takes 10 seconds)
results = [slow_function(i) for i in range(1, 11)]  # Run slow_function sequentially for numbers 1 through 10

# Execute the slow function with Ray (takes 1 second)
results_future = [slow_function_ray.remote(i) for i in range(1, 11)]  # Launch 10 Ray tasks in parallel and store their future object references
results_ray = ray.get(results_future)  # Retrieve the actual results from the Ray object references

print("Results without Ray: ", results)  # Print the results produced by the regular Python function
print("Results with Ray: ", results_ray)  # Print the results produced by the Ray remote function

ray.shutdown()  # Shut down Ray and release its resources