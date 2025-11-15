---
description: "Python coding conventions and guidelines"
applyTo: "**/*.py,**/*.ipynb"
---

# Python Coding Conventions

Your role is to act as a professional software engineer with deep expertise in Python and its ecosystem of libraries.

Your goal is to write clean, concise, and well-documented code that would be easy for others to maintain.
Variable naming should be self-documenting. Avoid the use of highly-nested and overly complex software.

Please use object oriented programming and encapsulate functionality into classes with common inheritance structures when relevant.
The code itself should be modular and easily extendable with a consistent API across the repository.

## Python Instructions

- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Provide docstrings following PEP 257 conventions.
- Use the `typing` module for type annotations (e.g., `List[str]`, `Dict[str, int]`).
- Break down complex functions into smaller, more manageable functions.

## General Instructions

- Always prioritize readability and clarity.
- For algorithm-related code, include explanations of the approach used.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- For libraries or external dependencies, mention their usage and purpose in comments.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.

## Code Style and Formatting

- Import standard libraries first, followed by third-party libraries, and then local application imports. Separate each group with a blank line and place all imports at the top of the file.
- Follow the **PEP 8** style guide for Python.
- Maintain proper indentation (use 4 spaces for each level of indentation).
- Ensure lines do not exceed 79 characters.
- Place function and class docstrings immediately after the `def` or `class` keyword.
- Use blank lines to separate functions, classes, and code blocks where appropriate.

## Naming Conventions

- Use PascalCase for component names, interfaces, and type aliases
- Use camelCase for variables, functions, and methods
- Prefix private class members with underscore (\_)
- Use ALL_CAPS for constants

## Documentation

- Give detailed documentation compatible with Sphinx documentation standards.
- Module docstrings should be in RST format. They must document all classes, functions, and global variables at a high level.
- Function and method docstrings should be in numpydoc format.
- Use inline comments to explain individual blocks of code.
- Type hinting should be used in all function and method signatures.

### Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
    """
    Calculate the area of a circle given the radius.

    Parameters:
    radius (float): The radius of the circle.

    Returns:
    float: The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * radius ** 2
```

## Edge Cases and Testing

- Always include test cases for critical paths of the application.
- Account for common edge cases like empty inputs, invalid data types, and large datasets.
- Include comments for edge cases and the expected behavior in those cases.
- Write unit tests for functions and document them with docstrings explaining the test cases.

## Library useage

- Use `logging` library to key code state information at the debug, info, warning, error, and ciritcal level to make monitoring applications and diagnosing issues easier.
- Use `typing` library to import various types for type hinting.
- Use `pytest` library for unit testing.
- Use `typer` for a beautiful command line interface. Options should have documentation explaining their use. The main command help string should also give various input examples.
- Use `mlflow` to register datasets and models as well as log metrics and model artifacts to quickly benchmark model performance.
- Use `torch` and `torch_geometric` for deep learning and GPU offloaded calculations. Use `torchmetrics` for calculating important metrics to quantify model performance.
- Use `chemprop` for property prediction and data preprocessing of chemistry and molecular datasets.
- Use `ray` for all hyper parameter optimization and batched simulation runs.
- Use `matplotlib` and `seaborn` with `colorcet` color schemes for all figures and tables.
- Use `numpy`, `scipy`, and `pandas` for basic mathematical transformations and data structures.
- Use `sklearn` for data preprocessing and basic machine learning.
- Use `xgboost` for gradient boosted decision tree models.
- Use `rdkit` for cheminformatics and molecular structure handling.
