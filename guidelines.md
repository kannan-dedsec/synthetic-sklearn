```markdown
# Guidelines for Sklearn Project

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style.
- Use 4 spaces per indentation level; do not use tabs.
- Limit lines to 79 characters for code and 72 characters for comments/docstrings.
- Use blank lines to separate functions and classes, and larger blocks of code inside functions.

## Import Organization

- Place imports at the top of the file, after any module comments and docstrings.
- Group imports in the following order:
  1. Standard library imports (e.g., `import os`, `import sys`)
  2. Related third-party imports (e.g., `import numpy as np`, `import pandas as pd`)
  3. Local application/library specific imports (e.g., `from my_module import my_function`)
- Use one import per line for clarity.
- Avoid wildcard imports (e.g., `from module import *`).

## Naming Conventions

- Use `snake_case` for functions and variable names (e.g., `train_model`, `data_frame`).
- Use `CamelCase` for class names (e.g., `DataPreprocessor`, `ModelEvaluator`).
- Use `UPPERCASE` for constants (e.g., `MAX_ITERATIONS`, `DEFAULT_ALPHA`).
- Use descriptive names that convey the purpose of the variable or function.
- Avoid single-character names except for counters or iterators (e.g., `i`, `j`).

## Documentation

- Use docstrings to describe all public modules, classes, methods, and functions.
- Follow the [Google style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings:
  - Use triple quotes for multi-line docstrings.
  - Include a summary line followed by a more detailed description if necessary.
  - Document parameters, return types, and exceptions raised.
  
  Example:
  ```python
  def train_model(X, y, model):
      """Train a machine learning model.

      Args:
          X (pd.DataFrame): Feature data.
          y (pd.Series): Target variable.
          model (sklearn.base.BaseEstimator): The model to train.

      Returns:
          sklearn.base.BaseEstimator: The trained model.
      """
  ```

## Common Pitfalls

### Mutable Defaults

- Avoid using mutable default arguments (e.g., lists or dictionaries) in function definitions.
- Instead, use `None` as the default and initialize the mutable object inside the function:
  
  ```python
  def process_data(data=None):
      """Process input data.

      Args:
          data (list, optional): List of data points. Defaults to None.
      """
      if data is None:
          data = []
      # Process data...
  ```

- This prevents unexpected behavior due to shared references across function calls.

## Conclusion

Adhering to these guidelines will help maintain code quality, readability, and consistency across the project. Regularly review and refactor code to align with these standards.
```
