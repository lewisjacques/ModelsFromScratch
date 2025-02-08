from typing import Callable
import re

### --- Primary Function Helper Class --- ###

class FunctionHelper:
    """
    Base function helper class.

    Handles wrappers for each of the main functional groups such as split_functions, 
    cost_functions etc. In the interest of longevity of this GitHub repository I believe
    storing functions in this way will help with the growth of the classes and functions 
    available.
    """

    def __init__(self):
        return
    
    @staticmethod
    def name_formatter(new_function:Callable):
        """
        Fall back function to double check all functions are named
        appropriately

        Args:
            new_function (Callable): as named
        """

        raw_function_name = new_function.__name__
        assert re.match(r"[a-z0-9_]+", raw_function_name), \
            "Naming convention of the base function methodology must be [a-z0-9_]"
        
        return(raw_function_name)
    
### --- Class For Decision Tree Splitting Functions --- ###

class SplitFunctionHelper(FunctionHelper):
    def __init__(self):
        self.all_split_functions = dict()

    def get_functions(self):
        return(self.all_split_functions)

    @staticmethod
    def name_formatter(new_split_function:Callable) -> str:
        """
        Overwrite naming convention for split functions

        Args:
            new_split_function (Callable): as named

        Returns:
            str: checked and formatted name
        """

        raw_function_name = new_split_function.__name__
        assert re.match(r"best_split_[a-z0-9_]+", raw_function_name), \
            "Naming convention of split functions must be best_split_xxx"
    
        return(raw_function_name)

    def add_split_function(
        self, 
        split_function:Callable
    ) -> Callable:
        """
        Wrapper for split_functions so we can dynamically import split
        functions

        Args:
            split_function (Callable): As named

        Returns:
            function: split_function once we've appended it's name to our tuple
        """

        # Extract the name and make sure naming conventions are consistent
        function_name = self.name_formatter(split_function)
        # Add function to the class dictionary
        self.all_split_functions[function_name] = split_function
        return
    
class CostFunctionHelper(FunctionHelper):
    def __init__(self):
        self.all_cost_functions = dict()

    def get_functions(self):
        return(self.all_cost_functions)

    def add_cost_function(
        self, 
        cost_function:Callable
    ) -> Callable:
        """
        Wrapper for cost_functions so we cna dynamically import and use
        cost functions without being explicit

        Args:
            cost_function (Callable): As named

        Returns:
            function: cost_function once we've appended it's name to our tuple
        """

        # Extract the name and make sure naming conventions are consistent
        function_name = self.name_formatter(cost_function)
        # Add function to the class dictionary
        self.all_cost_functions[function_name] = cost_function
        return