"""
This module leverages `pyprojroot` to dynamically locate the root directory of the project
and create paths relative to it. By using the `here()` function from `pyprojroot`, the
project root is automatically detected based on common root indicators (e.g., `.git/`,
`pyproject.toml`, or `setup.py`).

This enables consistent and portable path handling across the project, ensuring that
relative paths are resolved reliably, regardless of the script's location within the
project structure.

Dependencies:
-------------
- pyprojroot: Detects the root of the project.
- pathlib: Provides an object-oriented interface for handling filesystem paths.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

from pyprojroot import here


def requirements_path(filename: str = "requirements.txt") -> Path:
    """
    Return the path to a requirements file located at the project root.
    """
    return project_dir(filename)


def make_dir_function(dir_name: str | Iterable[str]) -> Callable[..., Path]:
    """
    Generate a function that constructs a path relative to the project directory,
    extending it with the provided subdirectory or subdirectories.

    Parameters
    ----------
    dir_name : Union[str, Iterable[str]]
        The name of the subdirectory (or a list of subdirectories) to append to the
        project root. If a single string is provided, it is treated as a single subdirectory.
        If an iterable of strings (e.g., a list) is provided, it will be joined into a path
        string, with separators dependent on the operating system.

    Returns
    -------
    Callable[..., Path]
        A function that, when called, returns the full path relative to the project directory.
        The returned function can accept additional arguments to further extend the path.
    """

    def dir_path(*args: str) -> Path:
        # Join the dir_name and any additional arguments as a path relative to the project root
        if isinstance(dir_name, str):
            return here().joinpath(dir_name, *args)
        return here().joinpath(*dir_name, *args)

    return dir_path


# Create the project directory function
project_dir = make_dir_function("")

# Define a comprehensive list of directory types
dir_types: list[list[str]] = [
    ["app"],  # API functionalities
    ["data"],  # Base data folder
    ["data", "raw"],  # Raw data folder
    ["data", "processed"],  # Processed data folder
    ["data", "interim"],  # Interim data folder
    ["data", "external"],  # External data folder (e.g., third-party sources)
    ["data", "figures"],  # Figures data folder
    ["models"],  # Folder to store models
    ["notebooks"],  # Jupyter notebooks folder
    ["references"],  # Reference materials
    ["reports"],  # Reports folder
    ["reports", "figures"],  # Figures for reports
    ["reports", "templates"],  # Templates for reports
    ["reports", "experiment_patients"],  # Experiment patients for reports
    ["tests"],  # Unit test files
    ["docs"],  # Documentation files
    ["logs"],  # Log files for running experiments or monitoring
    ["config"],  # Configuration files (e.g., YAML, JSON)
    ["scripts"],  # Standalone scripts (e.g., bash scripts, batch jobs)
]

# Use a dictionary to store dynamically created directory functions
dir_functions: dict[str, Callable[..., Path]] = {}

# Dynamically create directory functions and store them in the dictionary
for dir_type in dir_types:
    DIR_VAR_NAME = "_".join(dir_type) + "_dir"  # Create variable name like 'data_raw_dir'
    dir_functions[DIR_VAR_NAME] = make_dir_function(dir_type)

# Example usage:
# You can now access directories dynamically via the dir_functions dictionary
data_dir = dir_functions["data_dir"]
data_raw_dir = dir_functions["data_raw_dir"]
data_processed_dir = dir_functions["data_processed_dir"]
data_interim_dir = dir_functions["data_interim_dir"]
data_external_dir = dir_functions["data_external_dir"]
data_figures_dir = dir_functions["data_figures_dir"]
models_dir = dir_functions["models_dir"]
notebooks_dir = dir_functions["notebooks_dir"]
references_dir = dir_functions["references_dir"]
reports_dir = dir_functions["reports_dir"]
reports_figures_dir = dir_functions["reports_figures_dir"]
reports_templates_dir = dir_functions["reports_templates_dir"]
reports_experiment_patients_dir = dir_functions["reports_experiment_patients_dir"]
tests_dir = dir_functions["tests_dir"]
docs_dir = dir_functions["docs_dir"]
logs_dir = dir_functions["logs_dir"]
config_dir = dir_functions["config_dir"]
scripts_dir = dir_functions["scripts_dir"]
# # Example print statements to show directory paths
# print(f"Data Directory: {data_dir}")
# print(f"Raw Data Directory: {data_raw_dir}")
# print(f"Processed Data Directory: {data_processed_dir}")
# print(f"Models Directory: {models_dir}")
# print(f"Source Code Directory: {src_dir}")
# print(f"Scripts Directory: {scripts_dir}")
