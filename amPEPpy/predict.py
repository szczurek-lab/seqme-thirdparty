import sys
import pickle
import numpy as np
import pandas as pd

from typing import List

from pathlib import Path


def predict(sequences: List[str]) -> np.ndarray:
    repo_path = Path(__file__).resolve().parents[1]
    model_path = repo_path / "pretrained_models" / "amPEP.model"

    try:
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
    except FileNotFoundError:
        print(f"Model file: {model_path} not found!", file=sys.stderr)
        sys.exit(1)

    classify_df = score(sequences)

    preds = clf.predict_proba(classify_df)

    return preds[:, 1]


def score(sequences: List[str]) -> pd.DataFrame:
    CTD = {
        "hydrophobicity": {
            1: ["R", "K", "E", "D", "Q", "N"],
            2: ["G", "A", "S", "T", "P", "H", "Y"],
            3: ["C", "L", "V", "I", "M", "F", "W"],
        },
        "normalized.van.der.waals": {
            1: ["G", "A", "S", "T", "P", "D", "C"],
            2: ["N", "V", "E", "Q", "I", "L"],
            3: ["M", "H", "K", "F", "R", "Y", "W"],
        },
        "polarity": {
            1: ["L", "I", "F", "W", "C", "M", "V", "Y"],
            2: ["P", "A", "T", "G", "S"],
            3: ["H", "Q", "R", "K", "N", "E", "D"],
        },
        "polarizability": {
            1: ["G", "A", "S", "D", "T"],
            2: ["C", "P", "N", "V", "E", "Q", "I", "L"],
            3: ["K", "M", "H", "F", "R", "Y", "W"],
        },
        "charge": {
            1: ["K", "R"],
            2: [
                "A",
                "N",
                "C",
                "Q",
                "G",
                "H",
                "I",
                "L",
                "M",
                "F",
                "P",
                "S",
                "T",
                "W",
                "Y",
                "V",
            ],
            3: ["D", "E"],
        },
        "secondary": {
            1: ["E", "A", "L", "M", "Q", "K", "R", "H"],
            2: ["V", "I", "Y", "C", "W", "F", "T"],
            3: ["G", "N", "P", "S", "D"],
        },
        "solvent": {
            1: ["A", "L", "F", "C", "G", "I", "V", "W"],
            2: ["R", "K", "Q", "E", "N", "D"],
            3: ["M", "S", "P", "T", "H", "Y"],
        },
    }
    header = []
    groups = [1, 2, 3]
    values = [0, 25, 50, 75, 100]
    for AAproperty in CTD:
        for types in groups:
            for numbers in values:
                label = ""
                label = label.join("{}.{}.{}".format(AAproperty, types, numbers))
                header.append(label)
    all_groups = []
    sequence_names = []
    for i, sequence in enumerate(sequences):
        sequence_name = f"sequence_{i}"
        sequence_names.append(sequence_name)

        sequencelength = len(sequence)

        sequence_groups = []
        for AAproperty in CTD:
            propvalues = ""
            for letter in sequence:
                if letter in CTD[AAproperty][1]:
                    propvalues += "1"
                elif letter in CTD[AAproperty][2]:
                    propvalues += "2"
                elif letter in CTD[AAproperty][3]:
                    propvalues += "3"
            abpos_1 = [
                i for i in range(len(propvalues)) if propvalues.startswith("1", i)
            ]
            abpos_1 = [x + 1 for x in abpos_1]
            abpos_1.insert(0, "-")
            abpos_2 = [
                i for i in range(len(propvalues)) if propvalues.startswith("2", i)
            ]
            abpos_2 = [x + 1 for x in abpos_2]
            abpos_2.insert(0, "-")
            abpos_3 = [
                i for i in range(len(propvalues)) if propvalues.startswith("3", i)
            ]
            abpos_3 = [x + 1 for x in abpos_3]
            abpos_3.insert(0, "-")
            property_group1_length = propvalues.count("1")
            if property_group1_length == 0:
                sequence_groups.extend([0, 0, 0, 0, 0])
            elif property_group1_length == 1:
                sequence_groups.append((abpos_1[1] / sequencelength) * 100)
                sequence_groups.append((abpos_1[1] / sequencelength) * 100)
                sequence_groups.append((abpos_1[1] / sequencelength) * 100)
                sequence_groups.append((abpos_1[1] / sequencelength) * 100)
                sequence_groups.append((abpos_1[1] / sequencelength) * 100)
            elif property_group1_length == 2:
                sequence_groups.append((abpos_1[1] / sequencelength) * 100)
                sequence_groups.append((abpos_1[1] / sequencelength) * 100)
                sequence_groups.append(
                    (
                        abpos_1[round((0.5 * property_group1_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (
                        abpos_1[round((0.75 * property_group1_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (abpos_1[property_group1_length] / sequencelength) * 100
                )
            else:
                sequence_groups.append((abpos_1[1] / sequencelength) * 100)
                sequence_groups.append(
                    (
                        abpos_1[round((0.25 * property_group1_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (
                        abpos_1[round((0.5 * property_group1_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (
                        abpos_1[round((0.75 * property_group1_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (abpos_1[property_group1_length] / sequencelength) * 100
                )

            property_group2_length = propvalues.count("2")
            if property_group2_length == 0:
                sequence_groups.extend([0, 0, 0, 0, 0])
            elif property_group2_length == 1:
                sequence_groups.append((abpos_2[1] / sequencelength) * 100)
                sequence_groups.append((abpos_2[1] / sequencelength) * 100)
                sequence_groups.append((abpos_2[1] / sequencelength) * 100)
                sequence_groups.append((abpos_2[1] / sequencelength) * 100)
                sequence_groups.append((abpos_2[1] / sequencelength) * 100)
            elif property_group2_length == 2:
                sequence_groups.append((abpos_2[1] / sequencelength) * 100)
                sequence_groups.append((abpos_2[1] / sequencelength) * 100)
                sequence_groups.append(
                    (
                        abpos_2[round((0.5 * property_group2_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (
                        abpos_2[round((0.75 * property_group2_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (abpos_2[property_group2_length] / sequencelength) * 100
                )
            else:
                sequence_groups.append((abpos_2[1] / sequencelength) * 100)
                sequence_groups.append(
                    (
                        abpos_2[round((0.25 * property_group2_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (
                        abpos_2[round((0.5 * property_group2_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (
                        abpos_2[round((0.75 * property_group2_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (abpos_2[property_group2_length] / sequencelength) * 100
                )

            property_group3_length = propvalues.count("3")
            if property_group3_length == 0:
                sequence_groups.extend([0, 0, 0, 0, 0])
            elif property_group3_length == 1:
                sequence_groups.append((abpos_3[1] / sequencelength) * 100)
                sequence_groups.append((abpos_3[1] / sequencelength) * 100)
                sequence_groups.append((abpos_3[1] / sequencelength) * 100)
                sequence_groups.append((abpos_3[1] / sequencelength) * 100)
                sequence_groups.append((abpos_3[1] / sequencelength) * 100)
            elif property_group3_length == 2:
                sequence_groups.append((abpos_3[1] / sequencelength) * 100)
                sequence_groups.append((abpos_3[1] / sequencelength) * 100)
                sequence_groups.append(
                    (
                        abpos_3[round((0.5 * property_group3_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (
                        abpos_3[round((0.75 * property_group3_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (abpos_3[property_group3_length] / sequencelength) * 100
                )
            else:
                sequence_groups.append((abpos_3[1] / sequencelength) * 100)
                sequence_groups.append(
                    (
                        abpos_3[round((0.25 * property_group3_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (
                        abpos_3[round((0.5 * property_group3_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (
                        abpos_3[round((0.75 * property_group3_length) - 0.1)]
                        / sequencelength
                    )
                    * 100
                )
                sequence_groups.append(
                    (abpos_3[property_group3_length] / sequencelength) * 100
                )
        all_groups.append(sequence_groups)

    property_dataframe = pd.DataFrame.from_dict(all_groups)
    property_dataframe.columns = header
    property_dataframe.index = sequence_names

    return property_dataframe


if __name__ == "__main__":
    p_amp = predict(["KKKK", "RKRKRKRKR"])
    print(p_amp)
