"""
This module contains a class to connect to Mongo DB to extract data
and a class to extract data from storage blops
"""

import logging
import os
import time
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import requests
from azure.storage.blob import BlobServiceClient

# Create a logger
logger = logging.getLogger(__name__)


# def store_hdf5(
#     data: Dict[
#         str,
#         Union[
#             Dict[str, Dict[str, np.ndarray]],
#             Dict[str, np.ndarray],
#             List[Union[int, str]],
#             List[Union[float, int]],
#         ],
#     ],
#     path: Union[Path, str],
#     values_key: str,
#     features_key: str,
#     patients_key: str = "patient_id",
#     channel_key: str = "nodes",
#     add_creation_date: bool = True,
# ) -> None:
#     """
#     Stores a nested dictionary of features into an HDF5 file. This function writes
#     the nested structure of the data into an HDF5 file, making it easy to serialize
#     and deserialize impedance measurements and metadata.

#     Parameters
#     ----------
#     data : Dict[str, Union[
#             Dict[str, Dict[str, np.ndarray]],
#             Dict[str, np.ndarray],
#             List[Union[int, str]],
#             List[Union[float, int]]
#         ]]
#         A nested dictionary containing the data to store, structured as:
#             - `values_key`: A nested dictionary where:
#                 - The first key is "left" or "right" (laterality).
#                 - The second key is the BI-RADS category (e.g., "bi-rad 0").
#                 - The values are NumPy arrays with impedance measurements or feature values
#                   of shape (samples, nodes, features).
#             - `patient_id`: A dictionary mapping BI-RADS categories to lists of
#               patient IDs.
#             - `nodes`: A list of node (electrode) identifiers.
#             - `features_key`: A list of feature names corresponding to each column of
#               the feature data.
#     path : Union[Path, str]
#         Path to the HDF5 file where the data will be stored. Can be a string or a `Path` object.
#     values_key : str
#         The key in `data` that contains the nested dictionary of data
#         grouped by laterality and BI-RADS categories.
#     features_key : str
#         The key in `data` that contains the list of feature names.
#     patients_key : str, optional
#         The key in `data` that contains the list of patient IDs grouped by BI-RADS categories, by default "patient_id".
#     channel_key : str, optional
#         The key in `data` that contains the list of node (electrode) identifiers, by default "nodes".
#     add_creation_date : bool, optional
#         Whether to add the current date as a creation date in the HDF5 file, by default True

#     Raises
#     ------
#     ValueError
#         If any required key in the `data` dictionary is missing.
#     """
#     # Save data to an HDF5 file
#     with h5py.File(path, "w") as h5file:
#         # Store features values
#         features_values_group = h5file.create_group(values_key)
#         for side, bi_rads in data[values_key].items():
#             side_group = features_values_group.create_group(side)
#             for bi_rad, array in bi_rads.items():
#                 side_group.create_dataset(bi_rad, data=array)

#         # Store patient IDs
#         patient_id_group = h5file.create_group(patients_key)
#         for bi_rad, names in data[patients_key].items():
#             patient_id_group.create_dataset(bi_rad, data=np.array(names, dtype="S"))

#         # Store nodes
#         h5file.create_dataset(
#             channel_key,
#             data=np.array(data[channel_key], dtype="S"),
#             shape=(len(data[channel_key]),),
#         )

#         # Store feature names
#         h5file.create_dataset(
#             features_key,
#             data=np.array(data[features_key], dtype="S"),
#             shape=(len(data[features_key]),),
#         )

#         # Store creation date if required
#         if add_creation_date:
#             h5file.attrs["creation_date"] = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")


def store_hdf5(
    data: dict[
        str,
        dict[str, dict[str, np.ndarray]]
        | dict[str, np.ndarray]
        | list[int | str]
        | list[float | int],
    ],
    path: Path | str,
    values_key: str,
    features_key: str,
    patients_key: str = "patient_id",
    channel_key: str = "nodes",
    add_creation_date: bool = True,
    feature_level: str = None,
) -> None:
    """
    Store nested feature dictionaries into an HDF5 file.

    This version supports BOTH structures:

    ------------------------------------------------------------------------
    1) Node-level features
    ------------------------------------------------------------------------
    {
        "values": {
            "left": {
                "birad_1": ndarray(S, N, F),
                ...
            },
            "right": {
                ...
            }
        },
        "patient_id": {...},
        "nodes": [...],
        "features": [...]
    }

    ------------------------------------------------------------------------
    2) Aggregated/asymmetry features
    ------------------------------------------------------------------------
    {
        "values": {
            "left": {
                "birad_1": ndarray(S, F),
                ...
            },
            "right": {
                ...
            }
        },
        "patient_id": {...},
        "features": [...]
    }

    Parameters
    ----------
    data : dict
        Dictionary containing features and metadata.

    path : str or Path
        Output HDF5 file path.

    values_key : str
        Key containing feature arrays.

    features_key : str
        Key containing feature names.

    patients_key : str
        Key containing patient IDs.

    channel_key : str
        Optional key containing node/channel names.

    add_creation_date : bool
        Whether to store creation date.

    feature_level : str, optional
        Optional metadata:
            "node"
            "breast"
            "asymmetry"
            etc.
    """

    # =========================================================
    # Basic validation
    # =========================================================
    required_keys = [values_key, features_key, patients_key]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: '{key}'")

    # =========================================================
    # Create HDF5
    # =========================================================
    with h5py.File(path, "w") as h5file:
        # =====================================================
        # Store feature values
        # =====================================================
        values_group = h5file.create_group(values_key)

        values_data = data[values_key]

        if not isinstance(values_data, dict):
            raise ValueError(f"'{values_key}' must be a dictionary.")

        for key_1, value_1 in values_data.items():
            # -------------------------------------------------
            # CASE 1:
            # values -> {left/right -> bi-rad -> ndarray}
            # -------------------------------------------------
            if isinstance(value_1, dict):
                subgroup = values_group.create_group(key_1)

                for key_2, array in value_1.items():
                    if not isinstance(array, np.ndarray):
                        raise ValueError(
                            f"Expected ndarray at " f"{values_key}['{key_1}']['{key_2}']"
                        )

                    subgroup.create_dataset(
                        key_2,
                        data=array,
                        compression="gzip",
                    )

            # -------------------------------------------------
            # CASE 2:
            # values -> {bi-rad -> ndarray}
            # -------------------------------------------------
            elif isinstance(value_1, np.ndarray):
                values_group.create_dataset(
                    key_1,
                    data=value_1,
                    compression="gzip",
                )

            else:
                raise ValueError(f"Unsupported structure inside '{values_key}'.")

        # =====================================================
        # Store patient IDs
        # =====================================================
        patient_group = h5file.create_group(patients_key)

        patient_data = data[patients_key]

        if not isinstance(patient_data, dict):
            raise ValueError(f"'{patients_key}' must be a dictionary.")

        for group_name, patient_ids in patient_data.items():
            patient_group.create_dataset(
                group_name,
                data=np.array(patient_ids, dtype="S"),
            )

        # =====================================================
        # Store nodes/channels ONLY if present
        # =====================================================
        if channel_key in data:
            channels = data[channel_key]

            h5file.create_dataset(
                channel_key,
                data=np.array(channels, dtype="S"),
                shape=(len(channels),),
            )

        # =====================================================
        # Store feature names
        # =====================================================
        features = data[features_key]

        h5file.create_dataset(
            features_key,
            data=np.array(features, dtype="S"),
            shape=(len(features),),
        )

        # =====================================================
        # Metadata
        # =====================================================
        if add_creation_date:
            h5file.attrs["creation_date"] = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

        if feature_level is not None:
            h5file.attrs["feature_level"] = feature_level


# def load_hdf5(
#     path: Union[Path, str],
#     values_key: str,
#     features_key: str,
#     patients_key: str = "patient_id",
#     channel_key: str = "nodes",
#     date_key: str = None,
# ) -> Dict[
#     str,
#     Union[
#         Dict[str, Dict[str, np.ndarray]],
#         Dict[str, np.ndarray],
#         List[Union[int, str]],
#         List[Union[float, int]],
#     ],
# ]:
#     """
#     Loads data from an HDF5 file stored using the `store_hdf5` function.
#     This function deserializes the HDF5 file into a Python dictionary, preserving
#     the nested structure and metadata for further processing.

#     Parameters
#     ----------
#     path : Union[Path, str]
#         Path to the HDF5 file to load. Can be a string or a `Path` object.
#     values_key : str
#         The key in the HDF5 file corresponding to the nested dictionary
#         of data grouped by laterality and BI-RADS categories.
#     features_key : str
#         The key in the HDF5 file corresponding to the list of feature names.
#     patients_key : str
#         The key in the HDF5 file corresponding to the nested dictionary
#         of patient lists grouped by BI-RADS categories, by default "patient_id".
#     channel_key : str
#         The key in the HDF5 file corresponding to the list of node (electrode) identifiers, by default "nodes".
#     date_key : str, optional
#         The key in the HDF5 file corresponding to the creation date of the file, by default None.
#     Returns
#     -------
#     Dict[str, Union[
#         Dict[str, Dict[str, np.ndarray]],
#         Dict[str, np.ndarray],
#         List[Union[int, str]],
#         List[Union[float, int]]
#     ]]
#         A dictionary containing the loaded data with the following structure:
#             - `values_key`: A nested dictionary where:
#                 - The first key is "left" or "right" (laterality).
#                 - The second key is the BI-RADS category (e.g., "bi-rad 0").
#                 - The values are NumPy arrays of impedance measurements or feature values.
#             - `patient_id`: A dictionary mapping BI-RADS categories to lists
#               of patient IDs.
#             - `nodes`: A list of node (electrode) identifiers.
#             - `features_key`: A list of feature names corresponding to the columns
#               in the feature data.
#     """
#     with h5py.File(path, "r") as h5file:

#         # Load impedance data grouped by laterality and BI-RADS categories
#         features_values = {}
#         features_values_group = h5file[values_key]
#         for side in features_values_group:
#             features_values[side] = {}
#             side_group = features_values_group[side]
#             for bi_rad in side_group:
#                 features_values[side][bi_rad] = side_group[bi_rad][...]

#         # Load patient IDs
#         patient_id_group = h5file[patients_key]
#         patient_id = {
#             bi_rad: [name.decode("utf-8") for name in patient_id_group[bi_rad]]
#             for bi_rad in patient_id_group
#         }

#         # Load channels
#         channels = [node.decode("utf-8") for node in h5file[channel_key]]

#         # Load feature names
#         features = [feature.decode("utf-8") for feature in h5file[features_key]]

#         # Load creation date if date_key is provided
#         if date_key:
#             date = h5file.attrs[date_key]
#         else:
#             date = None

#     return {
#         values_key: features_values,
#         patients_key: patient_id,
#         channel_key: channels,
#         features_key: features,
#         date_key: date,
#     }


def load_hdf5(
    path: Path | str,
    values_key: str,
    features_key: str,
    patients_key: str = "patient_id",
    channel_key: str = "nodes",
    date_key: str = "creation_date",
) -> dict[
    str,
    dict[str, dict[str, np.ndarray]]
    | dict[str, np.ndarray]
    | list[int | str]
    | list[float | int]
    | str,
]:
    """
    Load data stored with `store_hdf5`.

    Supports BOTH:

    ------------------------------------------------------------------------
    1) Node-level features
    ------------------------------------------------------------------------
    {
        "values": {
            "left": {
                "birad_1": ndarray(S, N, F),
                ...
            },
            "right": {
                ...
            }
        },
        "patient_id": {...},
        "nodes": [...],
        "features": [...]
    }

    ------------------------------------------------------------------------
    2) Aggregated/asymmetry features
    ------------------------------------------------------------------------
    {
        "values": {
            "left": {
                "birad_1": ndarray(S, F),
                ...
            },
            "right": {
                ...
            }
        },
        "patient_id": {...},
        "features": [...]
    }

    Parameters
    ----------
    path : str or Path
        HDF5 file path.

    values_key : str
        Key containing feature arrays.

    features_key : str
        Key containing feature names.

    patients_key : str
        Key containing patient IDs.

    channel_key : str
        Optional key containing node/channel names.

    date_key : str
        Optional HDF5 attribute key for creation date.

    Returns
    -------
    dict
        Loaded feature dictionary.
    """

    out = {}

    with h5py.File(path, "r") as h5file:
        # =====================================================
        # Load feature values
        # =====================================================
        values_group = h5file[values_key]

        features_values = {}

        for key_1 in values_group.keys():
            item = values_group[key_1]

            # -------------------------------------------------
            # CASE 1:
            # values -> {left/right -> bi-rad -> ndarray}
            # -------------------------------------------------
            if isinstance(item, h5py.Group):
                features_values[key_1] = {}

                for key_2 in item.keys():
                    features_values[key_1][key_2] = item[key_2][...]

            # -------------------------------------------------
            # CASE 2:
            # values -> {bi-rad -> ndarray}
            # -------------------------------------------------
            elif isinstance(item, h5py.Dataset):
                features_values[key_1] = item[...]

            else:
                raise ValueError(f"Unsupported structure inside '{values_key}'.")

        out[values_key] = features_values

        # =====================================================
        # Load patient IDs
        # =====================================================
        patient_group = h5file[patients_key]

        patient_id = {}

        for group_name in patient_group.keys():
            patient_id[group_name] = [pid.decode("utf-8") for pid in patient_group[group_name][...]]

        out[patients_key] = patient_id

        # =====================================================
        # Load nodes/channels ONLY if present
        # =====================================================
        if channel_key in h5file:
            channels = [node.decode("utf-8") for node in h5file[channel_key][...]]

            out[channel_key] = channels

        # =====================================================
        # Load feature names
        # =====================================================
        features = [feature.decode("utf-8") for feature in h5file[features_key][...]]

        out[features_key] = features

        # =====================================================
        # Load creation date if available
        # =====================================================
        if date_key is not None and date_key in h5file.attrs:
            out[date_key] = h5file.attrs[date_key]

        # =====================================================
        # Load feature level metadata if available
        # =====================================================
        if "feature_level" in h5file.attrs:
            out["feature_level"] = h5file.attrs["feature_level"]

    return out


class MongoDBAuthClient:
    """
    A class to authenticate and retrieve data from MongoDB
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        email: str,
        password: str,
    ) -> None:
        """
        Initialize the client with API details and user credentials.

        Parameters
        ----------
        base_url : str
            Base URL of the API.
        api_key : str
            Key required for the API.
        email : str
            Email address for authentication.
        password : str
            Password for authentication.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.email = email
        self.password = password
        self.token = None
        self.token_expiry = 0  # Unix timestamp of token expiry
        self.auth_endpoint = "/users/login"  # Endpoint for login
        self.mamm_endpoint = "/datalake/mammography"  # Endpoint for mammography data
        self.test_endpoint = "/datalake/tests"  # Endpoint for test data
        self.cate_endpoint = "/datalake/categorical"  # Endpoint for categorical data
        self.users_endpoint = "/datalake/users"  # Endpoint for users data
        self.patients_endpoint = "/datalake/patients"  # Endpoint for patients data

        # /datalake/companies /datalake/locations # Other endpoints

    def authenticate(self) -> None:
        """
        Authenticate the user and retrieve the token.

        Raises
        ------
        Exception
            If authentication fails.
        """
        # API endpoint
        url = self.base_url + self.auth_endpoint

        # Required header
        headers = {"key": self.api_key}

        # Body payload
        payload = {"email": self.email, "password": self.password}

        # Make the POST request
        response = requests.post(url, json=payload, headers=headers, timeout=30)

        # Extract response data
        if response.status_code == 200:
            data = response.json()
            if "token" in data.get("data", {}):
                self.token = data["data"].get("token")
                # Set token to expire in 6 hours
                self.token_expiry = time.time() + 6 * 60 * 60
                logger.info("Authentication successful!")
            else:
                raise Exception("Authentication failed: Token not found in the response.")
        elif response.status_code == 401:
            raise Exception("Unauthorized: Check your API key, email, or password.")
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def get_token(self) -> str:
        """
        Get a valid token, refreshing it if expired.

        Returns:
            str: A valid authentication token.
        """
        if self.token is None or time.time() >= self.token_expiry:
            logger.info("Token expired or not available. Re-authenticating...")
            self.authenticate()
        return self.token

    def get_data(
        self,
        endpoint: Literal["mammography", "tests", "categoricals", "users", "patients"],
        query_params: str = None,
    ) -> list[dict[str, str | int | bool]]:
        """
        Fetch a batch of records from the given endpoint.

        Parameters
        ----------
        endpoint : str
            End point to retrieve data from.
        query_params : str, optional
            Query parameters to filter the data, by default None

        Returns
        -------
        List[Dict[str, Union[str, int, bool]]]
            List of records from the API response.

        Raises
        ------
        ValueError
            If invalid value for `endpoint`.
        Exception
            If the request fails or returns an error.
        """
        # Validate the `endpoint` argument
        valid_endpoints = ["mammography", "tests", "categoricals", "users", "patients"]
        if endpoint not in valid_endpoints:
            raise ValueError(
                f"Invalid value for `endpoint`: '{endpoint}'. "
                f"Expected one of {valid_endpoints}."
            )

        # Ensure the token is valid
        self.get_token()

        # Set endpoint
        if endpoint == "mammography":
            url = self.base_url + self.mamm_endpoint
        elif endpoint == "categoricals":
            url = self.base_url + self.cate_endpoint
        elif endpoint == "tests":
            url = self.base_url + self.test_endpoint
        elif endpoint == "users":
            url = self.base_url + self.users_endpoint
        elif endpoint == "patients":
            url = self.base_url + self.patients_endpoint

        # Set headers and params
        headers = {"token": self.token}
        params = {"query": query_params}

        # Make the GET request
        response = requests.get(url, headers=headers, params=params, timeout=30)

        # Extract response data
        if response.status_code == 200:
            data = response.json()
            logger.info("Data retrieval successful!")
            return data.get("data", [])  # Return the array of data
        elif response.status_code == 401:
            raise Exception("Unauthorized: Invalid token.")
        elif response.status_code == 500:
            raise Exception("Server error: Please try again later.")
        else:
            raise Exception(f"Request failed: {response.status_code} - {response.text}")

    def put_data(
        self,
        endpoint: Literal["mammography", "tests", "categoricals", "users", "patients"],
        query_params: str = None,
        body_params: str = None,
    ) -> None:
        """
        Update records from the given endpoint.

        Parameters
        ----------
        endpoint : str
            End point to update data from.
        query_params : str, optional
            Query parameters to filter the data, by default None.
        body_params : str, optional
            Data to update, by default None.

        Raises
        ------
        ValueError
            If invalid value for `endpoint`.
        Exception
            If the request fails or returns an error.
        """
        # Validate the `endpoint` argument
        valid_endpoints = ["mammography", "tests", "categoricals", "users", "patients"]
        if endpoint not in valid_endpoints:
            raise ValueError(
                f"Invalid value for `endpoint`: '{endpoint}'. "
                f"Expected one of {valid_endpoints}."
            )

        # Set endpoint
        if endpoint == "mammography":
            url = self.base_url + self.mamm_endpoint
        elif endpoint == "categoricals":
            url = self.base_url + self.cate_endpoint
        elif endpoint == "tests":
            url = self.base_url + self.test_endpoint
        elif endpoint == "users":
            url = self.base_url + self.users_endpoint
        elif endpoint == "patients":
            url = self.base_url + self.patients_endpoint

        # Set headers and params
        headers = {"token": self.token}
        body = body_params
        params = query_params  # Default query parameter

        # Make the PUT request
        response = requests.put(url, data=body, headers=headers, params=params, timeout=30)

        # Extract response data
        if response.status_code == 200:
            data = response.json()
            logger.info("Data retrieval successful!")
            return data.get("data", [])  # Return the array of data
        elif response.status_code == 401:
            raise Exception("Unauthorized: Invalid token.")
        elif response.status_code == 500:
            raise Exception("Server error: Please try again later.")
        else:
            raise Exception(f"Request failed: {response.status_code} - {response.text}")


class AzureBlobClient:
    """
    A class to manage Azure Blob Storage operations, optimized for
    batch downloading and uploading.
    """

    def __init__(self, connection_string: str | None = None) -> None:
        """
        Initialize the AzureBlobClient with an Azure Storage connection string.

        Parameters
        ----------
        connection_string : str, optional
            Azure Storage connection string for authentication. Required for operations.
        """
        if not connection_string:
            raise ValueError("A connection string must be provided.")
        self.connection_string = connection_string
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_name = None

    def set_container(self, container_name: str) -> None:
        """
        Set the default container name for batch operations.

        Parameters
        ----------
        container_name : str
            The name of the container to use for operations.
        """
        self.container_name = container_name

    def list_blobs(self, subdir: str | None = None) -> list[str]:
        """
        Lists the blobs in the specified container, optionally filtering by a subdirectory prefix.

        Parameters
        ----------
        subdir : Optional[str], optional
            The subdirectory prefix to filter the blobs. If None, all blobs in the container
            are listed. Defaults to None.

        Returns
        -------
        List[str]
            A list of blob names present in the specified container or subdirectory.

        Raises
        ------
        Exception
            If an error occurs while attempting to list the blobs.

        Notes
        -----
        - Blob storage is a flat namespace, but you can simulate a directory structure by including
        slashes ('/') in the blob names.
        - Ensure that the connection to the Azure Blob Storage account is properly configured
        in `self.blob_service_client` and `self.container_name`.
        """
        # Validate the container name
        if not self.container_name:
            raise ValueError("No container specified. Use 'set_container' to set a container.")

        try:
            # Get BlobClient for the container
            container_client = self.blob_service_client.get_container_client(self.container_name)

            # List blobs with an optional prefix
            blobs = (
                container_client.list_blobs(name_starts_with=subdir)
                if subdir
                else container_client.list_blobs()
            )

            # Extract blob names
            blob_names = [blob.name for blob in blobs]

            logger.info(
                f"Listed {len(blob_names)} blobs in container '{self.container_name}'"
                + (f" under subdirectory '{subdir}'" if subdir else "")
            )
            return blob_names

        except Exception as e:
            logger.error(f"Failed to list blobs in container '{self.container_name}/{subdir}': {e}")
            raise Exception(f"Failed to list blobs: {e}")

    def download_blob(self, blob_name: str, output_dir: Path | str) -> str:
        """
        Download a blob from the specified or default container.

        Parameters
        ----------
        blob_name : str
            The name of the blob to download.
        output_dir : Union[Path, str]
            Directory to save the downloaded file.

        Returns
        -------
        str
            The local file path of the downloaded blob.
        """
        if not self.container_name:
            raise ValueError("No container specified. Use 'set_container' to set a container.")

        try:
            # Get BlobClient for the blob
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )

            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Full local file path
            file_path = os.path.join(output_dir, os.path.basename(blob_name))

            # Download the blob to the file
            with open(file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            logger.info(f"Blob downloaded successfully: {file_path}")
            return file_path

        except Exception as e:
            raise Exception(f"Failed to download blob '{blob_name}': {e}")

    def download_blobs(self, blobs_names: list[str], output_dir: Path | str) -> list[str]:
        """
        Download multiple blobs from the specified or default container.

        Parameters
        ----------
        blobs_names : List[str]
            A list of blob names to download.
        output_dir : Union[Path, str]
            Directory to save the downloaded files.

        Returns
        -------
        List[str]
            A list of local file paths for the downloaded blobs.
        """
        if not self.container_name:
            raise ValueError("No container specified. Use 'set_container' to set a container.")

        downloaded_files = []
        for blob_name in blobs_names:
            local_path = self.download_blob(blob_name, output_dir)
            downloaded_files.append(local_path)
        return downloaded_files

    def upload_blob(self, container_subdir: str, file_path: Path | str) -> None:
        """
        Upload a file to the specified or default container.

        Parameters
        ----------
        container_subdir : str
            The name of the subdirectory in the container where the blops will be uploaded.
        file_path : Union[Path, str]
            The local file path to upload.
        """
        if not self.container_name:
            raise ValueError("No container specified. Use 'set_container' to set a container.")

        try:
            # Get BlobClient for the blob
            blob_name = f"{container_subdir}/{os.path.basename(file_path)}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )

            # Upload the file
            with open(file_path, "rb") as data:
                blob_client.upload_blob(
                    data=data, overwrite=True, blob_type="BlockBlob", max_concurrency=4
                )

            logger.info(f"Blob uploaded successfully: {blob_name}")

        except Exception as e:
            raise Exception(f"Failed to upload file '{file_path}' to blob '{blob_name}': {e}")

    def upload_blobs(
        self,
        container_subdir: str,
        files_paths: list[Path | str],
    ) -> None:
        """
        Upload multiple files to the specified or default container.

        Parameters
        ----------
        container_subdir : str
            The name of the subdirectory in the container where the blops will be uploaded.
        files_paths : List[Union[Path, str]]
            A list of local file paths to upload.
        """
        if not self.container_name:
            raise ValueError("No container specified. Use 'set_container' to set a container.")

        for file_path in files_paths:
            self.upload_blob(container_subdir, file_path)


if __name__ == "__main__":
    from julieta.credentials import (
        azure_storage_connection_string,
        mongodb_api_key,
        mongodb_password,
        mongodb_username,
    )
    from julieta.utils.paths import data_interim_dir

    # Instantiate client
    mongo_client = MongoDBAuthClient(
        base_url="https://testm.salvahealth.co/api/v1",
        api_key=mongodb_api_key,
        email=mongodb_username,
        password=mongodb_password,
    )

    # Authenticate the user
    mongo_client.get_token()

    # Fetch mammography records with a specific query
    try:
        mammo_records = mongo_client.get_data(endpoint="mammography")
        print(f"Number of fetched mammography records: {len(mammo_records)}")
    except Exception as e:
        print(f"Failed to fetch mammography records: {e}")

    # Fetch tests records with a specific query
    try:
        query = {"query": "testFile=false&calibrationFile=false"}  # Example query
        tests_records = mongo_client.get_data(endpoint="tests", query_params=query)
        print(f"Number of fetched measurements-info records: {len(tests_records)}")
    except Exception as e:
        print(f"Failed to fetch measurement information records: {e}")

    # Example list of blob names
    blop_names = [
        "66faf774ba12acd15c9aba6d_1732893269543.txt",
        "66fb006dba12acd15c9ac2a4_1732895030732.txt",
        "66fafeeeba12acd15c9abffc_1732894390892.txt",
    ]

    # Initialize the AzureBlopClient
    blop_client = AzureBlobClient(connection_string=azure_storage_connection_string)
    blop_client.set_container("prodfiles")

    # Download blops
    downloaded_files = []
    downloaded_files = blop_client.download_blobs(
        blobs_names=blop_names,
        output_dir=data_interim_dir("samples"),
    )

    blop_client.set_container("ai-julieta")
    blop_client.upload_blob(
        blob_name="raw-data/sample.h5",
        file_path=data_interim_dir("impedance_magnitude.h5"),
    )
