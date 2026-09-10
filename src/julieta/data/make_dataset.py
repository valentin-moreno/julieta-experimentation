from collections.abc import Callable
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd

from julieta.data.databases import load_hdf5
from julieta.features.compute_features import compute_statistics
from julieta.utils.paths import data_processed_dir, data_raw_dir


def make_train_validation_splitting_strategy(
    data: pd.DataFrame,
    birad_label_column: str,
    generate_synth: bool = False,
) -> tuple[dict[str, int], dict[str, int]]:
    samples_per_birad = data[birad_label_column].value_counts().to_dict()

    num_br1 = samples_per_birad["BI-RADS 1"]
    num_br2 = samples_per_birad["BI-RADS 2"]
    num_br3 = samples_per_birad["BI-RADS 3"]
    num_br4a = samples_per_birad["BI-RADS 4A"]
    num_br4b = samples_per_birad["BI-RADS 4B"]
    num_br4c = samples_per_birad["BI-RADS 4C"]
    num_br5 = samples_per_birad["BI-RADS 5"]
    sum_br2_br4b = num_br2 + num_br4b
    porc_br2 = round(num_br2 / sum_br2_br4b, 2)
    porc_br4b = round(num_br4b / sum_br2_br4b, 2)

    if generate_synth == False:
        sum_br3_br4a_br4c_br5 = num_br3 + num_br4a + num_br4c + num_br5
        quantity_br2_br4b = num_br1 - sum_br3_br4a_br4c_br5

        quantity_br2 = np.ceil(quantity_br2_br4b * porc_br2)
        quantity_br4b = np.floor(quantity_br2_br4b * porc_br4b)

        all_birads = (
            num_br1 + quantity_br2 + num_br3 + num_br4a + quantity_br4b + num_br4c + num_br5
        )
        quantity_train = np.ceil(all_birads * 0.8)
        quantity_test = np.floor(all_birads * 0.2)

        train_br1 = np.ceil(quantity_train / 2)
        train_br3 = np.ceil(num_br3 * 0.8)
        train_br4a = np.ceil(num_br4a * 0.8)
        train_br4c = np.ceil(num_br4c * 0.8)
        train_br5 = np.ceil(num_br5 * 0.8)

        train_sum_br3_br4a_br4c_br5 = train_br3 + train_br4a + train_br4c + train_br5
        train_br2_br4b = np.ceil(quantity_train / 2) - train_sum_br3_br4a_br4c_br5
        train_br2 = np.floor(train_br2_br4b * porc_br2)
        train_br4b = np.ceil(train_br2_br4b * porc_br4b)

        test_br1 = np.floor(quantity_test / 2)
        test_br3 = np.floor(num_br3 * 0.2)
        test_br4a = np.floor(num_br4a * 0.2)
        test_br4c = np.floor(num_br4c * 0.2)
        test_br5 = np.floor(num_br5 * 0.2)

        test_sum_br3_br4a_br4c_br5 = test_br3 + test_br4a + test_br4c + test_br5
        test_br2_br4b = np.ceil(quantity_test / 2) - test_sum_br3_br4a_br4c_br5
        test_br2 = np.floor(test_br2_br4b * porc_br2)
        test_br4b = np.ceil(test_br2_br4b * porc_br4b)

        num_samples_train = {
            "BI-RADS 1": int(train_br1),
            "BI-RADS 2": int(train_br2),
            "BI-RADS 3": int(train_br3),
            "BI-RADS 4A": int(train_br4a),
            "BI-RADS 4B": int(train_br4b),
            "BI-RADS 4C": int(train_br4c),
            "BI-RADS 5": int(train_br5),
        }

        num_samples_test = {
            "BI-RADS 1": int(test_br1),
            "BI-RADS 2": int(test_br2),
            "BI-RADS 3": int(test_br3),
            "BI-RADS 4A": int(test_br4a),
            "BI-RADS 4B": int(test_br4b),
            "BI-RADS 4C": int(test_br4c),
            "BI-RADS 5": int(test_br5),
        }
        strategy_synth = {
            "BI-RADS 1": 0,
            "BI-RADS 2": 0,
            "BI-RADS 3": 0,
            "BI-RADS 4A": 0,
            "BI-RADS 4B": 0,
            "BI-RADS 4C": 0,
            "BI-RADS 5": 0,
        }

    else:
        synth_br1 = np.ceil(num_br4a * 0.6 + num_br4c * 0.6 + num_br5 * 0.6)
        synth_br2 = np.ceil(num_br2 * 0)
        synth_br3 = np.ceil(num_br3 * 0)
        synth_br4a = np.ceil(num_br4a * 0.6)
        synth_br4b = np.ceil(num_br4b * 0)
        synth_br4c = np.ceil(num_br4c * 0.6)
        synth_br5 = np.ceil(num_br5 * 0.6)

        strategy_synth = {
            "BI-RADS 1": int(synth_br1),
            "BI-RADS 2": int(synth_br2),
            "BI-RADS 3": int(synth_br3),
            "BI-RADS 4A": int(synth_br4a),
            "BI-RADS 4B": int(synth_br4b),
            "BI-RADS 4C": int(synth_br4c),
            "BI-RADS 5": int(synth_br5),
        }

        total_nbr1 = synth_br1 + num_br1
        total_nbr3 = synth_br3 + num_br3
        total_nbr4a = synth_br4a + num_br4a
        total_nbr4c = synth_br4c + num_br4c
        total_nbr5 = synth_br5 + num_br5

        sum_br3_br4a_br4c_br5 = total_nbr3 + total_nbr4a + total_nbr4c + total_nbr5
        quantity_br2_br4b = total_nbr1 - sum_br3_br4a_br4c_br5

        quantity_br2 = np.ceil(quantity_br2_br4b * porc_br2)
        quantity_br4b = np.floor(quantity_br2_br4b * porc_br4b)

        n_total = (
            total_nbr1
            + quantity_br2
            + total_nbr3
            + total_nbr4a
            + quantity_br4b
            + total_nbr4c
            + total_nbr5
        )
        n_train = np.ceil(n_total * 0.8)
        n_test = np.floor(n_total * 0.2)

        test_br1 = np.floor(n_test / 2)
        test_br3 = np.floor(num_br3 * 0.2)
        test_br4a = np.floor(num_br4a * 0.2)
        test_br4c = np.floor(num_br4c * 0.2)
        test_br5 = np.floor(num_br5 * 0.2)

        test_sum_br3_br4a_br4c_br5 = test_br3 + test_br4a + test_br4c + test_br5
        test_br2_br4b = np.ceil(n_test / 2) - test_sum_br3_br4a_br4c_br5
        test_br2 = np.floor(test_br2_br4b * porc_br2)
        test_br4b = np.ceil(test_br2_br4b * porc_br4b)

        train_br1 = total_nbr1 - test_br1
        train_br3 = total_nbr3 - test_br3
        train_br4a = total_nbr4a - test_br4a
        train_br4c = total_nbr4c - test_br4c
        train_br5 = total_nbr5 - test_br5

        n_train_br2_br4b = n_train - train_br1 - train_br3 - train_br4a - train_br4c - train_br5
        train_br2 = np.ceil(n_train_br2_br4b * porc_br2)
        train_br4b = np.floor(n_train_br2_br4b * porc_br4b)

        num_samples_train = {
            "BI-RADS 1": int(train_br1 - synth_br1),
            "BI-RADS 2": int(train_br2 - synth_br2),
            "BI-RADS 3": int(train_br3 - synth_br3),
            "BI-RADS 4A": int(train_br4a - synth_br4a),
            "BI-RADS 4B": int(train_br4b - synth_br4b),
            "BI-RADS 4C": int(train_br4c - synth_br4c),
            "BI-RADS 5": int(train_br5 - synth_br5),
        }

        num_samples_test = {
            "BI-RADS 1": int(test_br1),
            "BI-RADS 2": int(test_br2),
            "BI-RADS 3": int(test_br3),
            "BI-RADS 4A": int(test_br4a),
            "BI-RADS 4B": int(test_br4b),
            "BI-RADS 4C": int(test_br4c),
            "BI-RADS 5": int(test_br5),
        }

    return num_samples_train, num_samples_test, strategy_synth


class MakeDataset:
    @staticmethod
    def train_val_splitting_strategy(
        df: pd.DataFrame,
        num_samples_train: dict[str, int],
        num_samples_val: dict[str, int],
        label_col: str,
        seed: int = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a DataFrame into training and validation sets based on per-class sampling strategies.

        This function creates a training set and a validation set by sampling a specified number of
        records from each class defined by `label_col`. The training set is formed by sampling
        `num_samples_train` records from each class after excluding the validation samples.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data to be split.
        num_samples_train : Dict[str, int]
            A dictionary defining the number of samples to include in the training set for each class.
            Keys are class labels (values of `label_col`), and values are the number of samples to
            include for that class.
        num_samples_val : Dict[str, int]
            A dictionary defining the number of samples to include in the validation set for each class.
            Keys are class labels (values of `label_col`), and values are the number of samples to
            include for that class.
        label_col : str
            The column name in `df` representing the class labels used to group the data.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing two DataFrames:
            - `train_set`: The training set containing the sampled records for training.
            - `val_set`: The validation set containing the sampled records for validation.

        Notes
        -----
        - If a class in `num_samples_train` or `num_samples_val` is not present in `df[label_col]`,
        it will be skipped without raising an error.
        - If a group contains fewer records than the combined total of `num_samples_train` and
        `num_samples_val`, a `ValueError` is raised.

        Raises
        ------
        ValueError
            If a class has fewer records than required to satisfy the sampling strategy.
            For example, if the total required samples for training and validation exceed the
            number of available records in the group.
        """
        # Initialize lists to store sampled training and validation data
        train_samples = []
        val_samples = []

        # Iterate over each class (group) defined by `label_col`
        for class_label, group in df.groupby(label_col):
            # Determine the number of records required for training and validation
            num_train = num_samples_train.get(class_label, 0)
            num_val = num_samples_val.get(class_label, 0)

            # Check if the group has enough records to satisfy the sampling strategy
            if len(group) < num_train + num_val:
                raise ValueError(
                    f"Class '{class_label}' has only {len(group)} records, but "
                    f"{num_train} training and {num_val} validation samples are required."
                )

            # Sample validation records
            if num_val > 0:
                if seed != None:
                    val_group = group.sample(n=num_val, random_state=seed)
                else:
                    val_group = group.sample(n=num_val)

                val_samples.append(val_group)
                # Exclude validation samples from the group for training
                group = group.drop(val_group.index)

            # Sample training records
            if num_train > 0:
                if seed != None:
                    train_group = group.sample(n=num_train, random_state=seed)
                else:
                    train_group = group.sample(
                        n=num_train,
                        # random_state=42
                    )
                train_samples.append(train_group)

        # Combine all sampled groups into training and validation sets
        train_set = pd.concat(train_samples) if train_samples else pd.DataFrame()
        val_set = pd.concat(val_samples) if val_samples else pd.DataFrame()

        return train_set, val_set

    @staticmethod
    def get_mammography_data(
        path: Path | str,
        patient_id_column: str = "patient_id",
        diagnosis_column: str = "birads_label",
        report: bool = False,
    ) -> pd.DataFrame:
        """
        Load and process mammography data.

        Supports both legacy and current Julieta datasets.
        """

        # Read data
        df = pd.read_csv(path)

        # Cast columns
        df[diagnosis_column] = df[diagnosis_column].astype(str).str.upper()
        df[patient_id_column] = df[patient_id_column].astype(str)

        # Normalize patient id column
        if patient_id_column != "patient_id":
            df = df.rename(columns={patient_id_column: "patient_id"})

        # -------------------------
        # Detect laterality column
        # -------------------------
        laterality_column = None

        if "laterality" in df.columns:
            laterality_column = "laterality"
        elif "relevantFindingLaterality" in df.columns:
            laterality_column = "relevantFindingLaterality"

        # Normalize laterality
        if laterality_column is not None:
            df[laterality_column] = (
                df[laterality_column]
                .astype(str)
                .str.lower()
                .replace(
                    {
                        "izquierda": "left",
                        "derecha": "right",
                        "ambas": "both",
                        "left": "left",
                        "right": "right",
                        "both": "both",
                    }
                )
            )

        if report:
            print("\n📌 Summary:")
            print(f"Total Records: {len(df)}")
            print(f"Unique Patients: {df['patient_id'].nunique()}")
            print(f"Diagnosis Missing: {df[diagnosis_column].isna().sum()}")

            if laterality_column:
                print(f"Laterality Missing: {df[laterality_column].isna().sum()}")
            else:
                print("Laterality Missing: N/A")

            report_columns = [
                "additionalFindingsInSameBreast",
                "additionalFindingsInSameBreastTypes",
                "anatomicalVariant",
                "asymmetryType",
                "breastDensity",
                "calcificationDistribution",
                "calcificationMorphology",
                "findingsInOtherBreast",
                "findingsInOtherBreastTypes",
                "mammogramReason",
                "mammogramType",
                "noduleContour",
                "noduleDensity",
                "noduleMorphology",
                "noduleSizeAnteroposterior",
                "noduleSizeLongitudinal",
                "noduleSizeTransverse",
                "relevantFindingQuadrant",
                "relevantFindingType",
            ]

            available_columns = [col for col in report_columns if col in df.columns]

            if available_columns:
                msno.matrix(
                    df[available_columns],
                    figsize=(20, 10),
                    color=(0.2, 0.4, 0.6),
                )
                plt.title("Missing Data Matrix")
                plt.show()

            if "breastDensity" in df.columns:
                plt.figure(figsize=(8, 4))
                df["breastDensity"].value_counts().plot(kind="bar")

                for i, v in enumerate(df["breastDensity"].value_counts()):
                    plt.text(i, v + 1, str(v), ha="center")

                plt.xlabel("Breast Density")
                plt.ylabel("Count")
                plt.title("Breast Density Distribution")
                plt.show()

            print("\n📊 Diagnosis Distribution:\n")

            if laterality_column:
                print(df.groupby([diagnosis_column, laterality_column]).size().rename("count"))
            else:
                print(
                    df[diagnosis_column].value_counts().rename_axis("Diagnosis").to_frame("count")
                )

            if "companyId" in df.columns:
                print("\n🏢 Company Distribution:\n")
                print(df["companyId"].value_counts().rename_axis("Company").to_frame("count"))

            if "location" in df.columns:
                print("\n📍 Location Distribution:\n")
                print(df["location"].value_counts().rename_axis("Location").to_frame("count"))

        return df

    @staticmethod
    def get_categoricals_data(
        path: Path | str,
        patient_id_column: str = "patient_id",
        company_id_column: str | None = None,
        report: bool = False,
    ) -> pd.DataFrame:
        # Read data
        df = pd.read_csv(path)

        # Cast columns
        df[patient_id_column] = df[patient_id_column].astype(str)

        if company_id_column is not None and company_id_column in df.columns:
            df[company_id_column] = df[company_id_column].astype(str)

        if report:
            print("\n📌 Summary:")
            print(f"Total Records: {len(df)}")
            print(f"Unique Patients: {df[patient_id_column].nunique()}")

            report_columns = [
                "adverseEventsDuringDevicePlacementOrTest",
                "age",
                "braCupSize",
                "breastRetractionPresence",
                "breastSecretionPresence",
                "createdBy",
                "eventsDescription",
                "familyCancerHistory",
                "familyCancerOther",
                "familyCancerTypes",
                "firstChildAge",
                "firstMammogramAge",
                "firstMenstruationAge",
                "height",
                "hormonalContraception",
                "hormonalTherapyTreatment",
                "irritationOrRednessPresence",
                "lastExamResult",
                "lastMenstrualCycleDay",
                "menopause",
                "menopauseAge",
                "numberOfChildren",
                "orangePeelSkinPresence",
                "painPresence",
                "personalCancerHistory",
                "personalCancerOther",
                "personalCancerTypes",
                "previousMammogramUltrasoundCT",
                "weight",
                "lastExamDate",
            ]

            available_columns = [col for col in report_columns if col in df.columns]

            if available_columns:
                msno.matrix(
                    df[available_columns],
                    figsize=(20, 10),
                    color=(0.2, 0.4, 0.6),
                )
                plt.title("Missing Data Matrix")
                plt.show()

            # Histogram helper
            for col, title in [
                ("age", "Age Distribution"),
                ("height", "Height Distribution"),
                ("weight", "Weight Distribution"),
            ]:
                if col in df.columns:
                    plt.figure(figsize=(8, 4))
                    plt.hist(
                        df[col].dropna(),
                        bins=20,
                        color="skyblue",
                        edgecolor="black",
                    )
                    plt.xlabel(col.capitalize())
                    plt.ylabel("Frequency")
                    plt.title(title)
                    plt.show()

            # Bra Cup Size
            if "braCupSize" in df.columns:
                plt.figure(figsize=(8, 4))
                counts = df["braCupSize"].value_counts()
                counts.plot(kind="bar")

                for i, v in enumerate(counts):
                    plt.text(i, v + 1, str(v), ha="center")

                plt.xlabel("Bra Cup Size")
                plt.ylabel("Count")
                plt.title("Bra Cup Size Distribution")
                plt.show()

            # Company Distribution
            if company_id_column is not None and company_id_column in df.columns:
                print("\n🏢 Company Distribution:\n")
                print(df[company_id_column].value_counts().rename_axis("Company").to_frame("count"))
            else:
                print("\n🏢 Company column not available.\n")

            # Location Distribution
            location_column = None

            if "location" in df.columns:
                location_column = "location"
            elif "locationId" in df.columns:
                location_column = "locationId"

            if location_column is not None:
                print("\n📍 Location Distribution:\n")
                print(df[location_column].value_counts().rename_axis("Location").to_frame("count"))
            else:
                print("\n📍 Location column not available.\n")

            # Recruitment Date
            if "recruitmentDate" in df.columns:
                print("\n📅 Recruitment Date:\n")
                print(df["recruitmentDate"].value_counts().rename_axis("Date").to_frame("count"))
            else:
                print("\n📅 Recruitment Date not available.\n")

        return df

    @staticmethod
    def get_measurements_data(
        output_data: Literal[
            "impedance_magnitude",
            "impedance_phase",
            "impedance",
            "resistance",
            "reactance",
            "admittance_magnitude",
            "admittance_phase",
            "admittance",
            "conductance",
            "susceptance",
            "m_imag",
        ],
        select_nodes: Literal[
            "all",
            "singulars",
            "opposites",
            "opposites_singulars",
            "adjacents",
            "adjacents_singulars",
            "l1",
            "l2",
            "l3",
            "upper",
            "lower",
        ],
        select_frequencies: str | list[str] | None = None,
        report: bool = False,
        study: str = "SURA",
        show_creation_date: bool = False,
    ) -> dict[
        str,
        dict[str, dict[str, np.ndarray]]
        | dict[str, np.ndarray]
        | list[int | str]
        | list[float | int],
    ]:
        """
        Retrieves and processes impedance data based on the specified output type,
        node selection criteria, and frequency selection.

        Parameters
        ----------
        output_data : Literal
            Specifies the type of impedance data to retrieve:
            - "impedance_magnitude": Impedance magnitude.
            - "impedance_phase": Impedance phase in degrees.
            - "impedance": Full complex impedance.
            - "resistance": Resistance.
            - "reactance": Reactance.
            - "admittance_magnitude": Admittance magnitude.
            - "admittance_phase": admittance phase in degrees.
            - "admittance": Full complex admittance.
            - "conductance": Conductance.
            - "susceptance": Susceptance.
            - "m_imag": Imag. part of the measurement

        select_nodes : Literal
            Specifies which nodes to include in the data:
            - "all": Include all nodes.
            - "singulars": Include selected singular nodes.
            - "opposites": Include opposite electrode pairs.
            - "opposites_singulars": Include a subset of opposite electrode pairs.
            - "adjacents": Includes adjacent electrode pairs.
            - "adjacents_singulars": Includes adjacent singular electrode pairs.
            - "l1": Nodes separated by the minimun distance.
            - "l2": Nodes separated by the medium distance.
            - "l3": Nodes separated by the maximum distance.
            - "upper": Upper nodes.
            - "lower": Lower nodes.

        select_frequencies : Optional[Union[str, List[str]]], optional
            Specifies frequencies to include in the data. If None, all frequencies are included.

        report : bool, optional
            If True, prints a summary report of the data structure after processing, by default False.

        study : str, optional
            The study name, by default "SURA"

        show_creation_date : bool, optional
            If True, prints the creation date of the .h5 file, by default False.
        Returns
        -------
        Dict
            A dictionary containing:
            - "measurements": Nested dictionary of measurements grouped by laterality
            ("left" or "right") and BI-RADS categories, with filtered nodes and frequencies.
            - "patient_id": Nested dictionary of patient IDs corresponding to the processed files.
            - "nodes": List of selected node labels.
            - "frequency_samples": List of selected or all frequencies.

        Raises
        ------
        ValueError
            If `output_data` is not one of the expected values.
            If `select_nodes` is not one of the expected values.
            If any `select_frequencies` values are invalid.
        """
        # Validate `output_data`
        valid_signals = [
            "impedance_magnitude",
            "impedance_phase",
            "impedance",
            "resistance",
            "reactance",
            "admittance_magnitude",
            "admittance_phase",
            "admittance",
            "conductance",
            "susceptance",
            "m_imag",
        ]
        if output_data not in valid_signals:
            raise ValueError(
                f"Invalid value for `output_data`: '{output_data}'. "
                f"Expected one of {valid_signals}."
            )

        # Validate `select_nodes`
        valid_nodes = [
            "all",
            "singulars",
            "opposites",
            "opposites_singulars",
            "adjacents",
            "adjacents_singulars",
            "l1",
            "l2",
            "l3",
            "upper",
            "lower",
        ]
        if select_nodes not in valid_nodes:
            raise ValueError(
                f"Invalid value for `select_nodes`: '{select_nodes}'. "
                f"Expected one of {valid_nodes}."
            )

        path = data_raw_dir(output_data + f"_{study}.h5")

        # Load data
        data_structure = load_hdf5(
            path,
            values_key="measurements",
            features_key="frequency_samples",
            date_key="creation_date",
        )

        # Define node selection
        node_selections = {
            "all": data_structure["nodes"],
            "singulars": [
                "12",
                "13",
                "14",
                "15",
                "16",
                "23",
                "24",
                "25",
                "26",
                "34",
                "35",
                "36",
                "45",
                "46",
                "56",
            ],
            "opposites": ["14", "25", "36", "41", "52", "63"],
            "opposites_singulars": ["14", "25", "36"],
            "adjacents": [
                "12",
                "23",
                "34",
                "45",
                "56",
                "16",
                "21",
                "32",
                "43",
                "54",
                "65",
                "61",
            ],
            "adjacents_singulars": ["12", "23", "34", "45", "56", "16"],
            "l1": [
                "12",
                "16",
                "21",
                "23",
                "32",
                "34",
                "43",
                "45",
                "54",
                "56",
                "61",
                "65",
            ],
            "l2": [
                "13",
                "15",
                "26",
                "24",
                "31",
                "35",
                "42",
                "46",
                "51",
                "53",
                "62",
                "64",
            ],
            "l3": ["14", "25", "36", "41", "52", "63"],
            "upper": [
                "46",
                "41",
                "36",
                "31",
            ],
            "lower": ["13", "14", "62", "64"],
        }
        nodes_selection = node_selections[select_nodes]

        # Get indices of selected nodes
        nodes_idx = [data_structure["nodes"].index(node) for node in nodes_selection]

        # Process frequency selection
        if select_frequencies is None:
            freq_idx = list(range(len(data_structure["frequency_samples"])))
        else:
            if isinstance(select_frequencies, str):
                select_frequencies = [select_frequencies]
            freq_idx = [
                data_structure["frequency_samples"].index(freq)
                for freq in select_frequencies
                if freq in data_structure["frequency_samples"]
            ]
            missing_freqs = set(select_frequencies) - set(data_structure["frequency_samples"])
            if missing_freqs:
                raise ValueError(f"Invalid frequencies: {missing_freqs}")

        # Filter measurements by selected nodes
        for laterality, data_dict in data_structure["measurements"].items():
            for bi_rad, data in data_dict.items():
                data_structure["measurements"][laterality][bi_rad] = data[:, nodes_idx, :][
                    :, :, freq_idx
                ]

        # Update nodes and frequency_samples in the data structure
        data_structure["nodes"] = nodes_selection
        data_structure["frequency_samples"] = [
            data_structure["frequency_samples"][i] for i in freq_idx
        ]

        # Show report if requested
        if report:
            print(f"\n Output data: {output_data} | Selected nodes: {select_nodes} \n")
            rows = []
            for side, data_dict in data_structure["measurements"].items():
                for bi_rad, data in data_dict.items():
                    m, n, k = data.shape
                    rows.append(
                        {
                            "Side": side.upper(),
                            "Study": bi_rad,
                            "Patients": m,
                            "Nodes": n,
                            "Frequencies": k,
                            "NaN Values": np.isnan(data).sum(),
                        }
                    )

            df_summary = pd.DataFrame(rows).set_index(["Side", "Study"])
            print(df_summary.sort_values(by=["Side", "Study"]).to_string())

        if show_creation_date:
            print("\n📅 Creation Date:", data_structure["creation_date"])

        return data_structure

    @staticmethod
    def get_features_data(
        path: Path | str,
        select_nodes: (
            Literal[
                "all",
                "singulars",
                "opposites",
                "opposites_singulars",
                "adjacents",
                "adjacents_singulars",
                "l1",
                "l2",
                "l3",
                "upper",
                "lower",
            ]
            | list[str]
        ) = "all",
        select_features: str | list[str] | None = None,
        values_key: str = "values",
        features_key: str = "features",
        channels_key: str = "nodes",
    ) -> dict[
        str,
        dict[str, dict[str, np.ndarray]]
        | dict[str, np.ndarray]
        | list[int | str]
        | list[float | int],
    ]:
        """
        Load and filter feature datasets.

        Supports ALL three feature structures:

        ------------------------------------------------------------------------
        1) Node-level features
        shape: (S, N, F)
        HAS nodes
        ------------------------------------------------------------------------

        2) Breast-level aggregated features
        shape: (S, F)
        NO nodes
        ------------------------------------------------------------------------

        3) Asymmetry features
        shape: (S, F)
        NO nodes
        ------------------------------------------------------------------------

        Node filtering is ONLY applied when node information exists.
        """

        # =========================================================
        # Load data
        # =========================================================
        data_structure = load_hdf5(
            path,
            values_key=values_key,
            features_key=features_key,
            channel_key=channels_key,
        )

        # =========================================================
        # Detect whether nodes exist
        # =========================================================
        has_nodes = channels_key in data_structure

        # =========================================================
        # Node selections
        # =========================================================
        node_selections = {
            "all": data_structure.get(channels_key, []),
            "singulars": [
                "12",
                "13",
                "14",
                "15",
                "16",
                "23",
                "24",
                "25",
                "26",
                "34",
                "35",
                "36",
                "45",
                "46",
                "56",
            ],
            "opposites": ["14", "25", "36", "41", "52", "63"],
            "opposites_singulars": ["14", "25", "36"],
            "adjacents": [
                "12",
                "23",
                "34",
                "45",
                "56",
                "16",
                "21",
                "32",
                "43",
                "54",
                "65",
                "61",
            ],
            "adjacents_singulars": ["12", "23", "34", "45", "56", "16"],
            "l1": [
                "12",
                "16",
                "21",
                "23",
                "32",
                "34",
                "43",
                "45",
                "54",
                "56",
                "61",
                "65",
            ],
            "l2": [
                "13",
                "15",
                "26",
                "24",
                "31",
                "35",
                "42",
                "46",
                "51",
                "53",
                "62",
                "64",
            ],
            "l3": ["14", "25", "36", "41", "52", "63"],
            "upper": ["46", "41", "36", "31"],
            "lower": ["13", "14", "62", "64"],
        }

        # =========================================================
        # Feature selection
        # =========================================================
        all_features = data_structure[features_key]

        if select_features is None:
            feature_idx = list(range(len(all_features)))

        else:
            if isinstance(select_features, str):
                select_features = [select_features]

            missing_features = set(select_features) - set(all_features)

            if missing_features:
                raise ValueError(f"Invalid features: {missing_features}")

            feature_idx = [all_features.index(feature) for feature in select_features]

        # =========================================================
        # Node selection (ONLY if nodes exist)
        # =========================================================
        if has_nodes:
            if isinstance(select_nodes, list):
                nodes_selection = select_nodes

            else:
                if select_nodes not in node_selections:
                    raise ValueError(f"Invalid node selection: {select_nodes}")

                nodes_selection = node_selections[select_nodes]

            missing_nodes = set(nodes_selection) - set(data_structure[channels_key])

            if missing_nodes:
                raise ValueError(f"Invalid nodes: {missing_nodes}")

            nodes_idx = [data_structure[channels_key].index(node) for node in nodes_selection]

        # =========================================================
        # Filter data
        # =========================================================
        for laterality, data_dict in data_structure[values_key].items():
            # -----------------------------------------------------
            # CASE 1:
            # laterality -> birad -> ndarray
            # -----------------------------------------------------
            if isinstance(data_dict, dict):
                for bi_rad, data in data_dict.items():
                    # ---------------------------------------------
                    # NODE-LEVEL
                    # shape: (S, N, F)
                    # ---------------------------------------------
                    if data.ndim == 3:
                        data = data[:, :, feature_idx]

                        if has_nodes:
                            data = data[:, nodes_idx, :]

                        data_structure[values_key][laterality][bi_rad] = data

                    # ---------------------------------------------
                    # AGGREGATED / ASYMMETRY
                    # shape: (S, F)
                    # ---------------------------------------------
                    elif data.ndim == 2:
                        data_structure[values_key][laterality][bi_rad] = data[:, feature_idx]

                    else:
                        raise ValueError(f"Unsupported ndarray shape: {data.shape}")

            # -----------------------------------------------------
            # CASE 2:
            # birad -> ndarray
            # -----------------------------------------------------
            elif isinstance(data_dict, np.ndarray):
                data = data_dict

                if data.ndim == 3:
                    data = data[:, :, feature_idx]

                    if has_nodes:
                        data = data[:, nodes_idx, :]

                elif data.ndim == 2:
                    data = data[:, feature_idx]

                else:
                    raise ValueError(f"Unsupported ndarray shape: {data.shape}")

                data_structure[values_key][laterality] = data

        # =========================================================
        # Update selected features
        # =========================================================
        data_structure[features_key] = [all_features[i] for i in feature_idx]

        # =========================================================
        # Update selected nodes ONLY if present
        # =========================================================
        if has_nodes:
            data_structure[channels_key] = nodes_selection

        return data_structure

    @staticmethod
    def aggregate_data_to_regions(
        data_structure: dict[
            str,
            dict[str, dict[str, np.ndarray]]
            | dict[str, list[str]]
            | list[int | str]
            | list[float | int],
        ],
        values_key: str,
        features_key: str,
        channels_key: str = "nodes",
        aggregation_funtion: Callable = np.mean,
        report: bool = False,
    ) -> dict[
        str,
        dict[str, dict[str, np.ndarray]]
        | dict[str, list[str]]
        | list[int | str]
        | list[float | int],
    ]:
        """
        Aggregate data to regions based on the specified aggregation function.

        Parameters
        ----------
        data_structure : Dict[
            str,
            Union[
                Dict[str, Dict[str, np.ndarray]],
                Dict[str, List[str]],
                List[Union[int, str]],
                List[Union[float, int]],
            ],
        ]
            The structured dataset containing:
            - `values_key`: A nested dictionary where the first key is the side
            (e.g., "left", "right"), the second key is BI-RADS category (e.g., "bi-rad 0"),
            and the values are NumPy arrays of shape (samples, channels, features).
            - `patient_id`: A dictionary mapping BI-RADS categories to lists of patient IDs.
            - `nodes`: A list of channel identifiers (e.g., nodes, electrodes, etc).
            - `features_key`: A list of feature names corresponding to measurements or signals.
        values_key : str
            The key in `data_structure` that contains the nested dictionary of data
            grouped by side and BI-RADS categories.
        features_key : str
            The key in `data_structure` that contains the list of feature names.
        channels_key : str, optional
            The key in `data_structure` that contains the list of channel identifiers, by default "nodes".
        aggregation_funtion : Callable, optional
            The aggegation function to use when aggregating the data nodes, by default np.mean
        report : bool, optional
            If True, prints a summary report of the data structure after processing, by default False.

        Returns
        -------
        Dict[
            str,
            Union[
                Dict[str, Dict[str, np.ndarray]],
                Dict[str, List[str]],
                List[Union[int, str]],
                List[Union[float, int]],
            ],
        ]
            A dictionary containing:
            - "values": Nested dictionary of values grouped by laterality
            ("left" or "right") and BI-RADS categories, with filtered nodes and frequencies.
            - "patient_id": Nested dictionary of patient IDs corresponding to the processed files.
            - "regions": List of regions.
            - "frequency_samples": List of selected or all frequencies.
        """
        # Define the nodes of the regiona
        regions = {
            "electrode1": ["12", "13", "14", "15", "16"],
            "electrode2": ["21", "23", "24", "25", "26"],
            "electrode3": ["31", "32", "34", "35", "36"],
            "electrode4": ["41", "42", "43", "45", "46"],
            "electrode5": ["51", "52", "53", "54", "56"],
            "electrode6": ["61", "62", "63", "64", "65"],
            "center": ["14", "25", "36", "41", "52", "63"],
        }

        # Allocat the data structure
        region_data = {
            side: {br: [] for br in item.keys()}
            for side, item in data_structure[values_key].items()
        }

        # Get the nodes names
        nodes = data_structure[channels_key]

        # Process data by side
        for side, data_dict in data_structure[values_key].items():
            # Process each BI-RADS group
            for bi_rad, data in data_dict.items():
                # Extract nodes for the region
                region_data_agg_list = []
                for _, region_nodes in regions.items():
                    # Get the nodes index for the region
                    nodes_idx = [nodes.index(node) for node in region_nodes]

                    # Apply the aggregation function to the region data
                    region_tensor = data[:, nodes_idx, :]  # shape: (samples, region_nodes, freqs)
                    region_data_agg = aggregation_funtion(
                        region_tensor, axis=1, keepdims=True
                    )  # shape: (samples, 1, freqs)

                    # Append the region data to the list
                    region_data_agg_list.append(region_data_agg)

                # Concatenate the region data along the channel axis
                region_data[side][bi_rad] = np.concatenate(region_data_agg_list, axis=1)

        # Structure computed features into a final output dictionary
        regions_data_structure = {
            "values": region_data,
            "patient_id": data_structure["patient_id"],
            "regions": list(regions.keys()),
            "frequency_samples": data_structure[features_key],
        }

        # Show report if requested
        if report:
            rows = []
            print(f"\n Region names: {list(regions.keys())} \n")
            for side, data_dict in regions_data_structure["values"].items():
                for bi_rad, data in data_dict.items():
                    m, n, k = data.shape
                    rows.append(
                        {
                            "Side": side.upper(),
                            "BI-RADS": bi_rad,
                            "Patients": m,
                            "Regions": n,
                            "Frequencies": k,
                            "NaN Values": np.isnan(data).sum(),
                        }
                    )

            df_summary = pd.DataFrame(rows).set_index(["Side", "BI-RADS"])
            print(df_summary.sort_values(by=["Side", "BI-RADS"]).to_string())

        return regions_data_structure

    @staticmethod
    def flatten_to_breast_representation(
        data_structure: dict[
            str,
            dict[str, dict[str, np.ndarray]]
            | dict[str, list[str]]
            | list[int | str]
            | list[float | int],
        ],
        values_key: str,
        features_key: str,
        channels_key: str = "nodes",
    ) -> pd.DataFrame:
        """
        Flatten a structured dataset into a tabular DataFrame with breast-level representation.

        Each row in the resulting DataFrame represents a patient breast record, with features
        flattened across channels and feature dimensions.

        Parameters
        ----------
        data_structure : Dict[
            str,
            Union[
                Dict[str, Dict[str, np.ndarray]],
                Dict[str, List[str]],
                List[Union[int, str]],
                List[Union[float, int]],
            ]
        ]
            The structured dataset containing:
            - `values_key`: A nested dictionary where the first key is the side
            (e.g., "left", "right"), the second key is BI-RADS category (e.g., "bi-rad 0"),
            and the values are NumPy arrays of shape (samples, channels, features).
            - `patient_id`: A dictionary mapping BI-RADS categories to lists of patient IDs.
            - `channels`: A list of channel identifiers (e.g., channels, electrodes, etc).
            - `features_key`: A list of feature names corresponding to measurements or signals.
        values_key : str
            The key in `data_structure` that contains the nested dictionary of data
            grouped by side and BI-RADS categories.
        features_key : str
            The key in `data_structure` that contains the list of feature names.
        channels_key : str, optional
            The key in `data_structure` that contains the list of channel identifiers, by default "nodes".

        Returns
        -------
        pd.DataFrame
            A flattened DataFrame where each row corresponds to a patient record, with:
            - Columns combining channels and features (e.g., "14_5000", "25_6000").
            - Additional columns for:
                - `patient_id`: Identifier for each patient.
                - `side`: "left" or "right" for each measurement.
                - `birads_label`: BI-RADS category label.
        """

        # Generate column names by combining channels and feature samples
        columns = [
            f"{channel}_{feature}"
            for channel in data_structure[channels_key]
            for feature in data_structure[features_key]
        ]

        side_data_list = []  # Holds the concatenated data for all sides
        patients_list = []  # Holds patient IDs
        side_list = []  # Holds side (e.g., "left" or "right")
        birads_labels = []  # Holds BI-RADS labels
        # Process data by side
        for side, data_dict in data_structure[values_key].items():
            # Process each BI-RADS group
            birads_data = []  # Holds data for all BI-RADS in the current side
            for bi_rad, data in data_dict.items():
                # Ensure data dimensions match expectations
                if data.shape[1] != len(data_structure[channels_key]) or data.shape[2] != len(
                    data_structure[features_key]
                ):
                    raise ValueError(
                        f"Data shape mismatch in side '{side}', BI-RADS '{bi_rad}'. "
                        f"Expected (samples, {len(data_structure[channels_key])}, "
                        f"{len(data_structure[features_key])}), got {data.shape}."
                    )

                # Extend metadata lists
                patients_list.extend(data_structure["patient_id"][bi_rad])
                side_list.extend([side] * data.shape[0])
                birads_labels.extend([bi_rad] * data.shape[0])

                # Flatten channel and frequency dimensions for each sample
                birads_data.append(data.reshape(data.shape[0], -1))

            # Concatenate data across BI-RADS categories for the current side
            side_data_list.append(np.concatenate(birads_data, axis=0))

        # Stack data across both sides
        stacked_data = np.concatenate(side_data_list, axis=0)

        # Create the DataFrame
        df = pd.DataFrame(stacked_data, columns=columns)
        df["patient_id"] = patients_list
        df["side"] = side_list
        df["birads_label"] = birads_labels
        df["birads_label"] = df["birads_label"].str.upper()

        # Set a multi-index for easier grouping/analysis
        df = df.set_index(["patient_id", "side"])

        return df

    @staticmethod
    def flatten_to_patient_representation(
        data_structure: dict[
            str,
            dict[str, dict[str, np.ndarray]]
            | dict[str, list[str]]
            | list[int | str]
            | list[float | int],
        ],
        values_key: str,
        features_key: str,
        channels_key: str = "nodes",
    ) -> pd.DataFrame:
        """
        Flatten a structured dataset into a tabular DataFrame with patient-level representation.

        Each row in the resulting DataFrame represents a patient record, with features flattened
        across channels, side and feature dimensions.

        Parameters
        ----------
        data_structure : Dict[
            str,
            Union[
                Dict[str, Dict[str, np.ndarray]],
                Dict[str, List[str]],
                List[Union[int, str]],
                List[Union[float, int]],
            ]
        ]
            The structured dataset containing:
            - `values_key`: A nested dictionary where the first key is the side
            (e.g., "left", "right"), the second key is BI-RADS category (e.g., "bi-rad 0"),
            and the values are NumPy arrays of shape (samples, channels, features).
            - `patient_id`: A dictionary mapping BI-RADS categories to lists of patient IDs.
            - `channels`: A list of channel identifiers (e.g., channels, electrodes, etc).
            - `features_key`: A list of feature names corresponding to measurements or signals.
        values_key : str
            The key in `data_structure` that contains the nested dictionary of data
            grouped by side and BI-RADS categories.
        features_key : str
            The key in `data_structure` that contains the list of feature names.

        Returns
        -------
        pd.DataFrame
            A flattened DataFrame where each row corresponds to a patient record, with:
            - Columns combining side, channels and features (e.g., "left_breast_14_5000", "left_breast_25_6000").
            - Additional columns for:
                - `patient_id`: Identifier for each patient.
                - `birads_label`: BI-RADS category label.
        """
        # Generate column names
        columns = [
            f"{side}_{channel}_{feature}"
            for side in data_structure[values_key].keys()
            for channel in data_structure[channels_key]
            for feature in data_structure[features_key]
        ]

        side_data_list = []  # Holds the concatenated data for all sides
        # Process data by side
        for side, data_dict in data_structure[values_key].items():
            # Process each BI-RADS group
            birads_data = []  # Holds data for all BI-RADS in the current side
            for bi_rad, data in data_dict.items():
                # Ensure data dimensions match expectations
                if data.shape[1] != len(data_structure[channels_key]) or data.shape[2] != len(
                    data_structure[features_key]
                ):
                    raise ValueError(
                        f"Data shape mismatch in side '{side}', BI-RADS '{bi_rad}'. "
                        f"Expected (samples, {len(data_structure[channels_key])}, "
                        f"{len(data_structure[features_key])}), got {data.shape}."
                    )

                # Flatten channel and frequency dimensions for each sample
                birads_data.append(data.reshape(data.shape[0], -1))

            # Concatenate data across BI-RADS categories for the current side
            side_data_list.append(np.concatenate(birads_data, axis=0))

        # Stack data across both sides
        stacked_data = np.concatenate(side_data_list, axis=1)

        # Create the DataFrame
        df = pd.DataFrame(stacked_data, columns=columns)
        df["patient_id"] = [id for ids in data_structure["patient_id"].values() for id in ids]
        df["birads_label"] = [
            br for keys, ids in data_structure["patient_id"].items() for br in [keys] * len(ids)
        ]
        df["birads_label"] = df["birads_label"].str.upper()

        # Set a multi-index for easier grouping/analysis
        df = df.set_index(["patient_id"])

        return df

    @staticmethod
    def flatten_to_channel_representation(
        data_structure: dict[
            str,
            dict[str, dict[str, np.ndarray]]
            | dict[str, list[str]]
            | list[int | str]
            | list[float | int],
        ],
        values_key: str,
        features_key: str,
        channels_key: str = "nodes",
    ) -> pd.DataFrame:
        """
        Flatten a structured dataset into a tabular DataFrame with channel-level representation.

        Each row in the resulting DataFrame represents a patient breast record, with features
        flattened across feature dimensions.

        Parameters
        ----------
        data_structure : Dict[
            str,
            Union[
                Dict[str, Dict[str, np.ndarray]],
                Dict[str, List[str]],
                List[Union[int, str]],
                List[Union[float, int]],
            ]
        ]
            The structured dataset containing:
            - `values_key`: A nested dictionary where the first key is the side
            (e.g., "left", "right"), the second key is BI-RADS category (e.g., "bi-rad 0"),
            and the values are NumPy arrays of shape (samples, channels, features).
            - `patient_id`: A dictionary mapping BI-RADS categories to lists of patient IDs.
            - `channels`: A list of channel identifiers (e.g., nodes, electrodes, etc).
            - `features_key`: A list of feature names corresponding to measurements or signals.
        values_key : str
            The key in `data_structure` that contains the nested dictionary of data
            grouped by side and BI-RADS categories.
        features_key : str
            The key in `data_structure` that contains the list of feature names.

        Returns
        -------
        pd.DataFrame
            A flattened DataFrame where each row corresponds to a patient record, with:
            - Column features (e.g., "5000", "6000").
            - Additional columns for:
                - `patient_id`: Identifier for each patient.
                - `side`: "left" or "right" for each measurement.
                - `birads_label`: BI-RADS category label.
                - `channel`: Channel identifier (e.g., "14", "25").
        """
        side_data_list = []  # Holds the concatenated data for all sides
        patients_list = []  # Holds patient IDs
        side_list = []  # Holds side (e.g., "left" or "right")
        channels_list = []  # Holds channel identifiers
        birads_labels = []  # Holds BI-RADS labels
        # Process data by side
        for side, data_dict in data_structure[values_key].items():
            # Process each BI-RADS group
            birads_data = []  # Holds data for all BI-RADS in the current side
            for bi_rad, data in data_dict.items():
                # Ensure data dimensions match expectations
                if data.shape[1] != len(data_structure[channels_key]) or data.shape[2] != len(
                    data_structure[features_key]
                ):
                    raise ValueError(
                        f"Data shape mismatch in side '{side}', BI-RADS '{bi_rad}'. "
                        f"Expected (samples, {len(data_structure[channels_key])}, "
                        f"{len(data_structure[features_key])}), got {data.shape}."
                    )

                # Extend metadata lists
                # patients_list.extend(data_structure["patient_id"][bi_rad] * data.shape[1])
                patients_list.extend(
                    np.repeat(
                        data_structure["patient_id"][bi_rad],
                        len(data_structure[channels_key]),
                    )
                )

                side_list.extend([side] * data.shape[0] * data.shape[1])
                channels_list.extend(data_structure[channels_key] * data.shape[0])
                birads_labels.extend([bi_rad] * data.shape[0] * data.shape[1])

                # Flatten channel and frequency dimensions for each sample
                birads_data.append(data.reshape(data.shape[0] * data.shape[1], -1))

            # Concatenate data across BI-RADS categories for the current side
            side_data_list.append(np.concatenate(birads_data, axis=0))

        # Stack data across both sides
        stacked_data = np.concatenate(side_data_list, axis=0)

        # Create the DataFrame
        df = pd.DataFrame(stacked_data, columns=data_structure[features_key])
        df["patient_id"] = patients_list
        df["side"] = side_list
        df["channel"] = channels_list
        df["channel"] = df["channel"].astype(str)
        df["birads_label"] = birads_labels
        df["birads_label"] = df["birads_label"].str.upper()

        # Set a multi-index for easier grouping/analysis
        df = df.set_index(["patient_id", "side", "channel"])

        return df

    @staticmethod
    def flatten_data(
        data_structure: dict[
            str,
            dict[str, dict[str, np.ndarray]]
            | dict[str, np.ndarray]
            | dict[str, list[str]]
            | list[int | str]
            | list[float | int],
        ],
        values_key: str,
        features_key: str,
        patient_id_key: str = "patient_id",
        channels_key: str = "nodes",
        data_representation: Literal["breast", "patient", "channel"] = "breast",
        drop_duplicates: bool = False,
        keep: Literal["first", "last", False] | None = False,
        drop_na: bool = False,
    ) -> pd.DataFrame:
        """
        Flatten datasets into a pandas DataFrame.

        Supports:
        ------------------------------------------------------------------------
        1) Node-level features
        shape: (S, N, F)
        WITH nodes
        ------------------------------------------------------------------------

        2) Breast-level aggregated features
        shape: (S, F)
        WITHOUT nodes
        ------------------------------------------------------------------------

        3) Asymmetry features
        shape: (S, F)
        WITHOUT nodes
        ------------------------------------------------------------------------
        """

        # =========================================================
        # Validate required keys
        # =========================================================
        required_keys = [
            values_key,
            features_key,
        ]

        for key in required_keys:
            if key not in data_structure:
                raise ValueError(f"Key '{key}' is missing in the data structure.")

        # =========================================================
        # Optional keys
        # =========================================================
        has_nodes = channels_key in data_structure
        has_patient_ids = patient_id_key in data_structure

        # =========================================================
        # Infer dimensionality
        # =========================================================
        sample_array = None

        for _, data_dict in data_structure[values_key].items():
            # CASE:
            # values[laterality][birad] = ndarray
            if isinstance(data_dict, dict):
                for _, arr in data_dict.items():
                    sample_array = arr
                    break

            # CASE:
            # values[birad] = ndarray
            elif isinstance(data_dict, np.ndarray):
                sample_array = data_dict

            if sample_array is not None:
                break

        if sample_array is None:
            raise ValueError("Could not infer dataset dimensionality.")

        # =========================================================
        # CASE 1: NODE-LEVEL FEATURES
        # shape: (S, N, F)
        # =========================================================
        if sample_array.ndim == 3:
            if not has_nodes:
                raise ValueError("3D node-level data requires 'nodes' key.")

            if data_representation == "breast":
                df = MakeDataset.flatten_to_breast_representation(
                    data_structure,
                    values_key=values_key,
                    features_key=features_key,
                    channels_key=channels_key,
                )

            elif data_representation == "patient":
                df = MakeDataset.flatten_to_patient_representation(
                    data_structure,
                    values_key=values_key,
                    features_key=features_key,
                    channels_key=channels_key,
                )

            elif data_representation == "channel":
                df = MakeDataset.flatten_to_channel_representation(
                    data_structure,
                    values_key=values_key,
                    features_key=features_key,
                    channels_key=channels_key,
                )

            else:
                raise ValueError(
                    "Invalid data_representation. " "Choose 'breast', 'patient', or 'channel'."
                )

        # =========================================================
        # CASE 2: BREAST-LEVEL / ASYMMETRY
        # shape: (S, F)
        # =========================================================
        elif sample_array.ndim == 2:
            rows = []

            # -----------------------------------------------------
            # FORMAT A:
            # values -> laterality -> birad -> array
            # -----------------------------------------------------
            first_value = next(iter(data_structure[values_key].values()))

            if isinstance(first_value, dict):
                for laterality, data_dict in data_structure[values_key].items():
                    for birad, data in data_dict.items():
                        # -----------------------------
                        # Patient IDs
                        # -----------------------------
                        if has_patient_ids:
                            patient_ids = data_structure[patient_id_key][birad]

                        else:
                            patient_ids = [f"sample_{i}" for i in range(data.shape[0])]

                        for sample_idx in range(data.shape[0]):
                            row = {
                                "laterality": laterality,
                                "birads_label": birad,
                            }

                            # patient_id optional
                            row[patient_id_key] = patient_ids[sample_idx]

                            # features
                            for feat_idx, feat_name in enumerate(data_structure[features_key]):
                                row[feat_name] = data[
                                    sample_idx,
                                    feat_idx,
                                ]

                            rows.append(row)

            # -----------------------------------------------------
            # FORMAT B:
            # values -> birad -> array
            # (ASYMMETRY FORMAT)
            # -----------------------------------------------------
            else:
                for birad, data in data_structure[values_key].items():
                    # -----------------------------
                    # Patient IDs
                    # -----------------------------
                    if has_patient_ids:
                        patient_ids = data_structure[patient_id_key][birad]

                    else:
                        patient_ids = [f"sample_{i}" for i in range(data.shape[0])]

                    for sample_idx in range(data.shape[0]):
                        row = {
                            "birads_label": birad,
                        }

                        # patient_id optional
                        row[patient_id_key] = patient_ids[sample_idx]

                        # features
                        for feat_idx, feat_name in enumerate(data_structure[features_key]):
                            row[feat_name] = data[
                                sample_idx,
                                feat_idx,
                            ]

                        rows.append(row)

            # -----------------------------------------------------
            # Build dataframe
            # -----------------------------------------------------
            df = pd.DataFrame(rows)

            # -----------------------------------------------------
            # Set index safely
            # -----------------------------------------------------
            index_cols = []

            if patient_id_key in df.columns:
                index_cols.append(patient_id_key)

            if "laterality" in df.columns:
                index_cols.append("laterality")

            if len(index_cols) > 0:
                df.set_index(index_cols, inplace=True)

        # =========================================================
        # Unsupported dimensions
        # =========================================================
        else:
            raise ValueError(f"Unsupported ndarray dimensionality: " f"{sample_array.ndim}")

        # =========================================================
        # Remove duplicates
        # =========================================================
        if drop_duplicates:
            duplicated_mask = df.index.duplicated(keep=keep)

            n_removed = duplicated_mask.sum()

            if n_removed > 0:
                print(f"\nRemoved {n_removed} duplicated index rows.")

                try:
                    print(
                        "\nNumber of duplicate unique patients removed: "
                        f"{df[duplicated_mask].index.get_level_values(patient_id_key).nunique()}"
                    )

                except Exception:
                    pass

                df = df[~duplicated_mask]

        # =========================================================
        # Drop NaNs
        # =========================================================
        if drop_na:
            print(f"\nBefore dropping NaN values, " f"the dataset has {df.shape[0]} samples.")

            df = df.dropna(axis=0)

            print(f"\nAfter dropping NaN values, " f"the dataset has {df.shape[0]} samples.")

        # =========================================================
        # Cleanup
        # =========================================================
        df.drop(
            columns=["birads_label"],
            inplace=True,
            errors="ignore",
        )

        return df

    def join_features_data(
        file_names: list[str],
        select_nodes: str = "all",
        node_statistics: bool = False,
    ) -> pd.DataFrame:
        """
        Loads and processes multiple feature datasets.

        Parameters
        ----------
        file_names : List[str]
            List of filenames to process.
        select_nodes : str, optional
            Nodes selection criteria, by default "all".
        node_statistics : bool, optional
            If True, computes statistical summaries of node data, by default False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the concatenated feature data.
        """

        features_df_list = []

        for file_name in file_names:
            data_dir = data_processed_dir(file_name)

            # Load the feature structure
            features_data_structure = MakeDataset.get_features_data(
                path=data_dir,
                select_nodes=select_nodes,
            )

            if node_statistics:
                # Compute statistics to reduce dimensionality
                features_data_structure = compute_statistics(
                    features_data_structure,
                    values_key="values",
                    axis=1,
                )
                channels_key = "statistics"
            else:
                channels_key = "nodes"

            # Flatten feature data
            features_data = MakeDataset.flatten_data(
                features_data_structure,
                values_key="values",
                features_key="features",
                channels_key=channels_key,
            )

            # Handle missing index columns safely
            index_cols = ["patient_id", "laterality", "birads_label"]
            existing_cols = [col for col in index_cols if col in features_data.columns]
            if existing_cols:
                features_data = features_data.set_index(existing_cols)

            # Append processed data to list
            features_df_list.append(features_data)

        # Ensure we return a valid DataFrame
        if not features_df_list:
            return pd.DataFrame()  # Return empty DataFrame if no files are processed

        return pd.concat(features_df_list, axis=1).reset_index()

    @staticmethod
    def filter_data_by_confirmed_laterality(
        df_data: pd.DataFrame,
        df_diagnosis: pd.DataFrame,
        laterality_column: str = "laterality",
    ) -> pd.DataFrame:
        """
        Creates a consolidated dataset by merging diagnostic data with measurement data
        based on laterality (e.g., left, right, or both).

        Parameters
        ----------
        df_data : pd.DataFrame
            The DataFrame containing measurement or feature data. It is expected to have:
            - `laterality`: A column indicating the breast side ("left_breast", "right_breast").
            - `patient_id`: A unique identifier for each patient.

        df_diagnosis : pd.DataFrame
            The DataFrame containing diagnostic information. It is expected to have:
            - `laterality_column`: A column indicating the breast side ("left", "right", "both").
            - `patient_id`: A unique identifier for each patient.

        laterality_column : str, optional
            The name of the column in `df_diagnosis` that specifies the laterality ("left", "right", "both").
            By default, it is "laterality".

        Returns
        -------
        pd.DataFrame
            A merged DataFrame indexed by `patient_id` and `laterality`, containing only records
            where measurement data matches the diagnostic data based on the laterality criteria.

        Raises
        ------
        KeyError
            If the required columns (`patient_id`, `laterality`) are missing from the input DataFrames.
        """
        # Get df_data indices names
        df_data_indices = df_data.index.names

        # Reset indices
        df_data = df_data.reset_index()

        # Perform separate merges for each laterality case and combine the results
        df = pd.concat(
            [
                # Merge left breast data with left laterality diagnosis
                df_data[df_data["side"] == "left_breast"].merge(
                    df_diagnosis[
                        (df_diagnosis[laterality_column] == "left")
                        | (df_diagnosis[laterality_column] == "izquierda")
                    ],
                    how="inner",
                    on="patient_id",
                ),
                # Merge right breast data with right laterality diagnosis
                df_data[df_data["side"] == "right_breast"].merge(
                    df_diagnosis[
                        (df_diagnosis[laterality_column] == "right")
                        | (df_diagnosis[laterality_column] == "derecha")
                    ],
                    how="inner",
                    on="patient_id",
                ),
                # Merge all data with "both" laterality diagnosis
                df_data.merge(
                    df_diagnosis[
                        (df_diagnosis[laterality_column] == "both")
                        | (df_diagnosis[laterality_column] == "ambas")
                    ],
                    how="inner",
                    on="patient_id",
                ),
            ]
        )
        df = df.set_index(df_data_indices)

        return df

    @staticmethod
    def train_validation_splitting(
        df: pd.DataFrame,
        num_samples_train: dict[str, int],
        num_samples_val: dict[str, int],
        label_col: str,
        seed: int = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a DataFrame into training and validation sets based on per-class sampling strategies.

        This function creates a training set and a validation set by sampling a specified number of
        records from each class defined by `label_col`. The training set is formed by sampling
        `num_samples_train` records from each class after excluding the validation samples.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data to be split.
        num_samples_train : Dict[str, int]
            A dictionary defining the number of samples to include in the training set for each class.
            Keys are class labels (values of `label_col`), and values are the number of samples to
            include for that class.
        num_samples_val : Dict[str, int]
            A dictionary defining the number of samples to include in the validation set for each class.
            Keys are class labels (values of `label_col`), and values are the number of samples to
            include for that class.
        label_col : str
            The column name in `df` representing the class labels used to group the data.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing two DataFrames:
            - `train_set`: The training set containing the sampled records for training.
            - `val_set`: The validation set containing the sampled records for validation.

        Notes
        -----
        - If a class in `num_samples_train` or `num_samples_val` is not present in `df[label_col]`,
        it will be skipped without raising an error.
        - If a group contains fewer records than the combined total of `num_samples_train` and
        `num_samples_val`, a `ValueError` is raised.

        Raises
        ------
        ValueError
            If a class has fewer records than required to satisfy the sampling strategy.
            For example, if the total required samples for training and validation exceed the
            number of available records in the group.
        """
        # Initialize lists to store sampled training and validation data
        train_samples = []
        val_samples = []

        # Iterate over each class (group) defined by `label_col`
        for class_label, group in df.groupby(label_col):
            # Determine the number of records required for training and validation
            num_train = num_samples_train.get(class_label, 0)
            num_val = num_samples_val.get(class_label, 0)

            # Check if the group has enough records to satisfy the sampling strategy
            if len(group) < num_train + num_val:
                raise ValueError(
                    f"Class '{class_label}' has only {len(group)} records, but "
                    f"{num_train} training and {num_val} validation samples are required."
                )

            # Sample validation records
            if num_val > 0:
                if seed != None:
                    val_group = group.sample(n=num_val, random_state=seed)
                else:
                    val_group = group.sample(n=num_val)
                val_samples.append(val_group)
                # Exclude validation samples from the group for training
                group = group.drop(val_group.index)

            # Sample training records
            if num_train > 0:
                if seed != None:
                    train_group = group.sample(n=num_train, random_state=seed)
                else:
                    train_group = group.sample(
                        n=num_train,
                        # random_state=42
                    )
                train_samples.append(train_group)

        # Combine all sampled groups into training and validation sets
        train_set = pd.concat(train_samples) if train_samples else pd.DataFrame()
        val_set = pd.concat(val_samples) if val_samples else pd.DataFrame()

        return train_set, val_set

    @staticmethod
    def downsample_training_data(
        X_train: pd.DataFrame, y_train: pd.Series, sampling_strategy: dict[str, int]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Downsample by selecting a specified number of samples from each category (birad).

        Parameters
        ----------
        X_train : pd.DataFrame
            The input DataFrame containing the data to be downsampled.
        y_train : pd.Series
            The labels Series.
        sampling_strategy : Dict[str, int]
            A dictionary specifying the number of samples to select from each birad category.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Downsampled X_train and y_train.
        """
        # List to store the downsampled indices
        sampled_indices = []

        # Iterate over each class in y_train
        for label, group in y_train.groupby(y_train):
            if label in sampling_strategy:  # Ensure the label is in the strategy dict
                n_samples = min(sampling_strategy[label], len(group))  # Avoid errors
                sampled_indices.extend(group.sample(n_samples, random_state=42).index)

        # Select the downsampled data
        X_train_downsampled = X_train.loc[sampled_indices]
        y_train_downsampled = y_train.loc[sampled_indices]

        return X_train_downsampled, y_train_downsampled
