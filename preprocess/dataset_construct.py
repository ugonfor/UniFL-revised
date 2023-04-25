import pandas as pd
import numpy as np
import os
import warnings
from collections import Counter
import argparse

warnings.filterwarnings("ignore")

# !IMPORTANT
# We Already have a Label, And Already have a filtered dataset!
# So I remove all the filtering process and labeling related process.


# Create MIMIC-III dataset
def create_mimiciii_ICU(args):
    time_gap_hours = args.time_gap_hours
    pred_window_hours = args.pred_window_hours
    timegap = pd.Timedelta(time_gap_hours, unit="h")
    pred_window = pd.Timedelta(pred_window_hours, unit="h")

    patient_path = os.path.join(args.rawdata_path, "mimiciii", "PATIENTS.csv")
    icustay_path = os.path.join(args.rawdata_path, "mimiciii", "ICUSTAYS.csv")
    # dx_path = os.path.join(args.rawdata_path, "mimiciii", "DIAGNOSES_ICD.csv")
    ad_path = os.path.join(args.rawdata_path, "mimiciii", "ADMISSIONS.csv")

    patients = pd.read_csv(patient_path)
    icus = pd.read_csv(icustay_path)
    ad = pd.read_csv(ad_path)
    # dx = pd.read_csv(dx_path)

    print("length of PATIENTS.csv  : ", len(patients))
    print("length of ICUSTAYS.csv  : ", len(icus))
    # print("length of DIAGNOSIS_ICD.csv  : ", len(dx))
    print("length of ADMISSION.csv  : ", len(ad))


    #icus = icus.drop(columns=["ROW_ID"])
    icus["INTIME"] = pd.to_datetime(icus["INTIME"], infer_datetime_format=True)
    icus["OUTTIME"] = pd.to_datetime(icus["OUTTIME"], infer_datetime_format=True)

    patients["DOB"] = pd.to_datetime(patients["DOB"], infer_datetime_format=True)
    #patients = patients.drop(columns=["ROW_ID"])

    small_patients = patients[patients.SUBJECT_ID.isin(icus.SUBJECT_ID)]
    icus = icus.merge(small_patients, on="SUBJECT_ID", how="left")


    readmit = icus.groupby("HADM_ID")["ICUSTAY_ID"].count()
    readmit_labels = (
        (readmit > 1)
        .astype("int64")
        .to_frame()
        .rename(columns={"ICUSTAY_ID": "readmission"})
    )
    print("readmission value counts :", readmit_labels.value_counts())
    print("[!] If there is readmission, Use have to careful!")
    print("[!] In our filtered case, There is no readmit HADM")
    cohort = icus

    print(cohort.info())

    inputdata_path = os.path.join(args.inputdata_path, "mimiciii_cohort.pkl")
    print(f"The final mimiciii cohort pickle is saved at: {inputdata_path}")
    cohort.to_pickle(inputdata_path)


# Create eICU dataset
def create_eICU_ICU(args):

    time_gap_hours = args.time_gap_hours
    pred_window_hours = args.pred_window_hours
    timegap = pd.Timedelta(time_gap_hours, unit="h")
    pred_window = pd.Timedelta(pred_window_hours, unit="h")

    patient_path = os.path.join(args.rawdata_path, "eicu", "patient.csv")
    patient_df = pd.read_csv(patient_path)

    print("Unique patient unit stayid : ", len(set(patient_df.patientunitstayid)))

    micu = patient_df

    readmit = micu.groupby("patienthealthsystemstayid")["patientunitstayid"].count()
    readmit_labels = (
        (readmit > 1)
        .astype("int64")
        .to_frame()
        .rename(columns={"patientunitstayid": "readmission"})
    )
    print("readmission value counts :", readmit_labels.value_counts())
    print("[!] If there is readmission, Use have to careful!")
    print("[!] In our filtered case, There is no readmit HADM")


    inputdata_path = os.path.join(args.inputdata_path, "eicu_cohort.pkl")
    print(f"The final eicu cohort pickle is saved at: {inputdata_path}")
    micu.to_pickle(inputdata_path)


# Create mimiciv dataset
def create_mimiciv_ICU(args):
    time_gap_hours = 2
    pred_window_hours = 24
    timegap = pd.Timedelta(time_gap_hours, unit="h")
    pred_window = pd.Timedelta(pred_window_hours, unit="h")

    icu = pd.read_csv(os.path.join(args.rawdata_path, "mimiciv", "icustays.csv"))
    adm = pd.read_csv(os.path.join(args.rawdata_path, "mimiciv", "admissions.csv"))
    pat = pd.read_csv(os.path.join(args.rawdata_path, "mimiciv", "patients.csv"))
    
    def columns_upper(df):
        df.columns = [x.upper() for x in df.columns]

    for df in [icu, pat, adm]:
        columns_upper(df)
        
    # ICU
    icu.rename(columns={"STAY_ID": "ICUSTAY_ID"}, inplace=True)

    df = icu[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]].groupby("HADM_ID").count()
    df["readmission"] = df["ICUSTAY_ID"].apply(lambda x: 1 if x > 1 else 0)
    print(df["readmission"].value_counts())

    icu.sort_values("INTIME", ascending=True, inplace=True)

    icu = icu.groupby("HADM_ID").first()
    icu["HADM_ID"] = icu.index
    icu.reset_index(drop=True, inplace=True)

    cohort = icu
    inputdata_path = os.path.join(args.inputdata_path, "mimiciv_cohort.pkl")
    print(f"The final mimiciv cohort pickle is saved at: {inputdata_path}")
    cohort.to_pickle(inputdata_path)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rawdata_path", type=str, required=True)
    parser.add_argument("--inputdata_path", type=str, required=True)
    parser.add_argument("--event_window_hours", type=int, default=12)
    parser.add_argument("--time_gap_hours", type=int, default=12)
    parser.add_argument("--pred_window_hours", type=int, default=48)
    parser.add_argument(
        "--icu_type", type=str, choices=["micu", "ticu", "ticu_multi"], default="ticu"
    )
    return parser


def main():
    args = get_parser().parse_args()
    print("create mimiciii ICU start!")
    create_mimiciii_ICU(args)
    print("create eICU ICU start!")
    create_eICU_ICU(args)
    print("create mimiciv ICU start!")
    create_mimiciv_ICU(args)


if __name__ == "__main__":
    main()
