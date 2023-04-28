import argparse

import pandas as pd
import numpy as np
import os
import warnings
from collections import Counter
import argparse

from transformers import AutoTokenizer
import swifter
import argparse

warnings.filterwarnings("ignore")

class my_args:
    def __init__(self, rawdata_path, inputdata_path) -> None:
        self.rawdata_path = rawdata_path
        self.inputdata_path = inputdata_path
        self.event_window_hours = 12
        self.time_gap_hours = 12
        self.pred_window_hours = 48
        self.sample = False

CONFIG = {
    "Table": {
        "mimiciii": [
            {
                "table_name": "LABEVENTS",
                "time_column": "CHARTTIME",
                "table_type": "lab",
                "time_excluded": [],
                "id_excluded": [
                    "ROW_ID",
                    "SUBJECT_ID"
                ]
            },
            {
                "table_name": "PRESCRIPTIONS",
                "time_column": "STARTDATE",
                "table_type": "med",
                "time_excluded": [
                    "ENDDATE"
                ],
                "id_excluded": [
                    "GSN",
                    "NDC",
                    "ROW_ID",
                    "HADM_ID",
                    "SUBJECT_ID"
                ]
            },
            {
                "table_name": "INPUTEVENTS_CV",
                "time_column": "CHARTTIME",
                "table_type": "inf",
                "time_excluded": [
                    "ENDTIME",
                    "STORETIME",
                    "COMMENTS_DATE"
                ],
                "id_excluded": [
                    "CGID",
                    "ORDERID",
                    "LINKORDERID",
                    "ROW_ID",
                    "HADM_ID",
                    "SUBJECT_ID"
                ]
            },
            {
                "table_name": "INPUTEVENTS_MV",
                "time_column": "STARTTIME",
                "table_type": "inf",
                "time_excluded": [
                    "ENDTIME",
                    "STORETIME",
                    "COMMENTS_DATE"
                ],
                "id_excluded": [
                    "CGID",
                    "ORDERID",
                    "LINKORDERID",
                    "ROW_ID",
                    "HADM_ID",
                    "SUBJECT_ID"
                ]
            }
        ],
        "eicu": [
            {
                "table_name": "lab",
                "time_column": "labresultoffset",
                "table_type": "lab",
                "time_excluded": [
                    "labresultrevisedoffset"
                ],
                "id_excluded": [
                    "labid"
                ]
            },
            {
                "table_name": "medication",
                "time_column": "drugstartoffset",
                "table_type": "med",
                "time_excluded": [
                    "drugorderoffset",
                    "drugstopoffset"
                ],
                "id_excluded": [
                    "medicationid",
                    "gtc",
                    "drughiclseqno"
                ]
            },
            {
                "table_name": "infusionDrug",
                "time_column": "infusionoffset",
                "table_type": "inf",
                "time_excluded": [],
                "id_excluded": [
                    "infusiondrugid"
                ]
            }
        ],
        "mimiciv": [
            {
                "table_name": "labevents",
                "time_column": "CHARTTIME",
                "table_type": "lab",
                "time_excluded": [
                    "STORETIME"
                ],
                "id_excluded": [
                    "SUBJECT_ID",
                    "SPECIMEN_ID",
                    "LABEVENT_ID"
                ]
            },
            {
                "table_name": "prescriptions",
                "time_column": "STARTTIME",
                "table_type": "med",
                "time_excluded": [
                    "STOPTIME"
                ],
                "id_excluded": [
                    "GSN",
                    "NDC",
                    "SUBJECT_ID",
                    "PHARMACY_ID"
                ]
            },
            {
                "table_name": "inputevents",
                "time_column": "STARTTIME",
                "table_type": "inf",
                "time_excluded": [
                    "ENDTIME",
                    "STORETIME"
                ],
                "id_excluded": [
                    "ORDERID",
                    "LINKORDERID",
                    "SUBJECT_ID"
                ]
            }
        ]
    },
    "selected": {
        "mimiciii": {
            "LABEVENTS": {
                "ID": "ID",
                "ITEMID": "code",
                "VALUENUM": "value",
                "VALUEUOM": "uom"
            },
            "PRESCRIPTIONS": {
                "ID": "ID",
                "DRUG": "code",
                "ROUTE": "route",
                "PROD_STRENGTH": "prod",
                "DOSE_VAL_RX": "value",
                "DOSE_UNIT_RX": "uom"
            },
            "INPUTEVENTS_CV": {
                "ID": "ID",
                "ITEMID": "code",
                "RATE": "value",
                "RATEUOM": "uom"
            }
        },
        "eicu": {
            "lab": {
                "ID": "ID",
                "labname": "code",
                "labresult": "value",
                "labmeasurenamesystem": "uom"
            },
            "medication": {
                "ID": "ID",
                "drugname": "code",
                "routeadmin": "route",
                "value": "value",
                "uom": "uom"
            },
            "infusionDrug": {
                "ID": "ID",
                "drugname": "code",
                "infusionrate": "value",
                "uom": "uom"
            }
        },
        "mimiciv": {
            "labevents": {
                "ID": "ID",
                "ITEMID": "code",
                "VALUENUM": "value",
                "VALUEUOM": "uom"
            },
            "prescriptions": {
                "ID": "ID",
                "DRUG": "code",
                "PROD_STRENGTH": "prod",
                "DOSE_VAL_RX": "value",
                "DOSE_UNIT_RX": "uom"
            },
            "inputevents": {
                "ID": "ID",
                "ITEMID": "code",
                "RATE": "value",
                "RATEUOM": "uom"
            }
        }
    },
    "DICT_FILE": {
        "mimiciii": {
            "LABEVENTS": [
                "D_LABITEMS",
                "ITEMID"
            ],
            "INPUTEVENTS_CV": [
                "D_ITEMS",
                "ITEMID"
            ]
        },
        "eicu": {},
        "mimiciv": {
            "labevents": [
                "d_labitems",
                "ITEMID"
            ],
            "inputevents": [
                "d_items",
                "ITEMID"
            ]
        }
    },
    "ID": {
        "mimiciii": "ICUSTAY_ID",
        "eicu": "patientunitstayid",
        "mimiciv": "STAY_ID"
    },
    "SUB_ID": {
        "mimiciii": "HADM_ID",
        "mimiciv": "HADM_ID"
    }
}

NUMERIC_DICT = {
    "mimiciii": {
        "LABEVENTS": {
            "value": [
                "VALUE",
                "VALUENUM"
            ],
            "cate": [],
            "code": "ITEMID"
        },
        "PRESCRIPTIONS": {
            "value": [
                "DOSE_VAL_RX",
                "FORM_VAL_DISP"
            ],
            "cate": [],
            "code": "DRUG"
        },
        "INPUTEVENTS": {
            "value": [
                "AMOUNT",
                "RATE",
                "PATIENTWEIGHT",
                "TOTALAMOUNT",
                "ORIGINALAMOUNT",
                "ORIGINALRATE"
            ],
            "cate": [
                "ISOPENBAG",
                "CONTINUEINNEXTDEPT",
                "CANCELREASON"
            ],
            "code": "ITEMID"
        }
    },
    "eicu": {
        "lab": {
            "value": [
                "labresult",
                "labresulttext"
            ],
            "cate": [
                "labtypeid"
            ],
            "code": "labname"
        },
        "medication": {
            "value": [
                "value"
            ],
            "cate": [
                "drugordercancelled",
                "drugivadmixture"
            ],
            "code": "drugname"
        },
        "infusionDrug": {
            "value": [
                "drugrate",
                "infusionrate",
                "drugamount",
                "volumeoffluid",
                "patientweight"
            ],
            "cate": [],
            "code": "drugname"
        }
    },
    "mimiciv": {
        "labevents": {
            "value": [
                "VALUE",
                "VALUENUM",
                "REF_RANGE_UPPER",
                "REF_RANGE_LOWER"
            ],
            "cate": [],
            "code": "ITEMID"
        },
        "prescriptions": {
            "value": [
                "DOSE_VAL_RX",
                "FORM_VAL_DISP",
                "DOSES_PER_24_HRS"
            ],
            "cate": [],
            "code": "DRUG"
        },
        "inputevents": {
            "value": [
                "AMOUNT",
                "RATE",
                "PATIENTWEIGHT",
                "TOTALAMOUNT",
                "ORIGINALAMOUNT",
                "ORIGINALRATE"
            ],
            "cate": [
                "ISOPENBAG",
                "CONTINUEINNEXTDEPT",
                "CANCELREASON",
                "VALUECOUNTS"
            ],
            "code": "ITEMID"
        }
    }
}
# !IMPORTANT
# We Already have a Label, And Already have a filtered dataset!
# So I remove all the filtering process and labeling related process.

#####
# fold split
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import warnings

warnings.filterwarnings("ignore")


def random_split(df, seed, test_split, val_split):
    df.loc[:, f"{seed}_rand"] = 1

    df_train, df_test = train_test_split(
        df, test_size=1 / test_split, random_state=seed, shuffle=True
    )
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_test.loc[:, f"{seed}_rand"] = 0

    df_train, df_valid = train_test_split(
        df_train, test_size=1 / val_split, random_state=seed, shuffle=True
    )
    df_valid.loc[:, f"{seed}_rand"] = 2
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)

    df = pd.concat([df_train, df_valid, df_test], axis=0)
    df.sort_values(by="pid", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def stratified(df, col, seed, test_split, val_split):
    split = ShuffleSplit(
        n_splits=1, test_size=1 / test_split, random_state=seed
    )
    # train / test
    for train_idx, test_idx in split.split(df, df[col]):
        df_strat_test = df.loc[test_idx].reset_index(drop=True)
        df_strat_train = df.loc[train_idx].reset_index(drop=True)
        df_strat_test[col + f"_{seed}_strat"] = 0

    split = ShuffleSplit(
        n_splits=1, test_size=1 / val_split, random_state=seed
    )
    
    for train_idx, val_idx in split.split(df_strat_train, df_strat_train[col]):
        df_strat_valid = df_strat_train.loc[val_idx].reset_index(drop=True)
        df_strat_train = df_strat_train.loc[train_idx].reset_index(drop=True)
        df_strat_valid[col + f"_{seed}_strat"] = 2

    df = pd.concat([df_strat_train, df_strat_valid, df_strat_test], axis=0)
    df.sort_values(by="pid", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def multiclass_multilabel_stratified(df, col, seed):
    
    msss = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for train_index, test_index in msss.split(
        np.array(df["pid"].tolist()), np.array(df["dx"].tolist())
    ):
        print("TRAIN:", train_index, "TEST:", test_index)
        df.loc[test_index, col + f"_{seed}_strat"] = 0

        break
    df_test = df[df[col + f"_{seed}_strat"] == 0].reset_index(drop=True)
    df_train = df[df[col + f"_{seed}_strat"] == 1].reset_index(drop=True)

    msss = MultilabelStratifiedKFold(n_splits=9, shuffle=True, random_state=seed)

    for train_index, valid_index in msss.split(
        np.array(df_train["pid"].tolist()), np.array(df_train["dx"].tolist())
    ):
        print("TRAIN:", train_index, "valid:", valid_index)
        df_train.loc[valid_index, col + f"_{seed}_strat"] = 2
        break

    df_train.reset_index(drop=True, inplace=True)
    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    return df


def stratified_split(df, seed, test_split, val_split):
    col = "labels"
    print("columns : ", col)
    if col in df.columns:
        df[col + f"_{seed}_strat"] = 1
        df = stratified(df, col, seed, test_split, val_split)
    else:
        raise AssertionError("Wrong, check!")

    return df

#####


######
# preprocess utils
import pandas as pd
import numpy as np
import re
from operator import itemgetter
from itertools import groupby

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

quant=20
vocab = {
    '[PAD]' : 0,
    '[CLS]' : 1,
    '[SEP]' : 2,
    '[MASK]' : 3
}

vocab['TB_0'] = 4
start_idx = 5
for qb in range(1, quant+1):
    vocab[f'TB_{qb}'] = start_idx
    start_idx+=1

number_token_list = [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 119] 


def eicu_med_revise(df):
    df['split'] = df['dosage'].apply(lambda x: str(re.sub(',', '',str(x))).split())
    def second(x):
        try:
            if len(pd.to_numeric(x))>=2:
                x = x[1:]
            return x
        except ValueError:
            return x

    df['split'] = df['split'].apply(second).apply(lambda s:' '.join(s))
    punc_dict = str.maketrans('', '', '.-')
    df['uom'] = df['split'].apply(lambda x: re.sub(r'[0-9]', '', x))
    df['uom'] = df['uom'].apply(lambda x: x.translate(punc_dict)).apply(lambda x: x.strip())
    df['uom'] = df['uom'].apply(lambda x: ' ' if x=='' else x)
    
    def hyphens(s):
        if '-' in str(s):
            s = str(s)[str(s).find("-")+1:]
        return s
    df['value'] = df['split'].apply(hyphens)
    df['value'] = df['value'].apply(lambda x: [float(s) for s in re.findall(r'-?\d+\.?\d*', x)])
    df['value'] = df['value'].apply(lambda x: x[-1] if len(x)>0 else x)
    df['value'] = df['value'].apply(lambda d: str(d).replace('[]',' '))
    df = df.drop('split',axis=1)
    df = df.drop('dosage',axis=1)
    return df


def eicu_inf_revise(df):
    df['split'] = df['drugname'].apply(lambda x: str(x).rsplit('(', maxsplit=1))
    def addp(x):
        if len(x)==2:
            x[1] = '(' + str(x[1])
        return x

    df['split'] = df['split'].apply(addp)
    df['split']=df['split'].apply(lambda x: x +[' '] if len(x)<2 else x)

    df['drugname'] = df['split'].apply(lambda x: x[0])
    df['uom'] = df['split'].apply(lambda x: x[1])
    df['uom'] = df['uom'].apply(lambda s: s[s.find("("):s.find(")")+1])

    toremove = ['()','', '(Unknown)', '(Scale B)', '(Scale A)',  '(Human)', '(ARTERIAL LINE)']

    df['uom'] = df['uom'].apply(lambda uom: ' ' if uom in toremove else uom)
    df = df.drop('split',axis=1)
    
    testing = lambda x: (str(x)[-1].isdigit()) if str(x)!='' else False
    code_with_num = list(pd.Series(df.drugname.unique())[pd.Series(df.drugname.unique()).apply(testing)==True])
    add_unk = lambda s: str(s)+' [UNK]' if s in code_with_num else s
    df['drugname'] = df['drugname'].apply(add_unk)
    
    return df


def name_dict(df, code_dict, column_name):
    key = code_dict['ITEMID']
    value = code_dict['LABEL']
    code_dict = dict(zip(key,value))
    df[column_name] = df[column_name].map(code_dict)
    df[column_name] = df[column_name].map(str)
    
    return df


def ID_time_filter_eicu(df, icu):
   
    df = df[df['ID'].isin(icu['ID'])] # 'ID' filter
    time_fil= df[(df['TIME'] > 0) &
                      (df['TIME'] < 60*12)
                ]
    return time_fil


def ID_time_filter_mimic(df, icu):
    df = df[df['ID'].isin(icu['ID'])]# ID filter
    df['ID'] = df['ID'].astype('int')
    df['TIME'] = pd.to_datetime(df['TIME'])
  
    df = df.merge(icu[['ID', 'INTIME']], on='ID', how='left').reset_index(drop=True)
    df['TIME'] = (df['TIME'] - df['INTIME']).astype('timedelta64[m]')
    df.drop(columns=['INTIME'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def columns_upper(df):
    df.columns = [x.upper() for x in df.columns]
    return


def codeemb_event_merge(df, table_name):

    target_cols =  [ col for col in df.columns if col not in ['ID','TIME', 'time_bucket', 'time_gap', 'TABLE_NAME', 'ORDER']]
    
    df['event'] = df.apply(lambda x: [x[col] for col in target_cols if x[col] != ' '], axis=1)
    df['type'] = df.apply(lambda x: [table_name+ '_' + col for col in target_cols if x[col] !=' '], axis=1)

    df['event_token'] = df.apply(lambda x : x['event'] + [x['time_bucket']], axis=1)
    df['type_token'] = df.apply(lambda x : x['type'] + ['[TIME]'], axis=1)               
   
    return df


def buckettize_categorize(df, src, numeric_dict, table_name, quant):
    code = numeric_dict[src][table_name]['code']

    type_code = df[code].dtype
    for value_target in numeric_dict[src][table_name]['value']: 
        if value_target in df.columns:
            numeric = df[pd.to_numeric(df[value_target], errors='coerce').notnull()]
            numeric[value_target]= numeric[value_target].astype('float')
            not_numeric = df[pd.to_numeric(df[value_target], errors='coerce').isnull()]

            # buckettize
            numeric = buckettize(numeric, code, value_target, quant)


            numeric[value_target] = 'B_' + numeric[value_target].astype('str')
            df = pd.concat([numeric, not_numeric], axis=0)

    for cate_target in numeric_dict[src][table_name]['cate']:
        if cate_target in df.columns:
            df = categoritize(df, cate_target)
    df[code] = df[code].astype(type_code)

    df.fillna(' ', inplace=True)
    df.replace('nan', ' ', inplace=True)
    return df


def make_dpe(target, number_token_list, integer_start=6):
    if type(target) is not list:
        return None
    elif target ==[]:
        return []
    else:
        dpe = [1]*len(target) # dpe token 1 for the plain text token
        scanning = [pos for pos, char in enumerate(target) if char in number_token_list]

        #grouping
        ranges = []
        for k,g in groupby(enumerate(scanning),lambda x:x[0]-x[1]):
            group = (map(itemgetter(1),g))
            group = list(map(int,group))
            ranges.append((group[0],group[-1]))

        # making dpe     
        dpe_group_list = []
        for (start, end) in ranges:
            group = target[start:end+1]
            digit_index = [pos for pos, char in enumerate(group) if char == number_token_list[-1]] #digit_token
            assert len(digit_index) < 3, "More than 3 digit index in sing group"
            if len(digit_index)==2:
                # ex) 1. 0 2 5. 
                if digit_index[0] == 0:
                    group= group[1:]
                    digit_index = digit_index[1:]
                    start=start+1
            # case seperate if digit or integer only
            if len(digit_index)== 0:
                dpe_group = [integer_start+len(group)-i for i in range(len(group))]
            else:
                # 있으면 소수점 기준으로 왼쪽 오른 walk
                dpe_int = [integer_start-1+len(group[:digit_index[0]])-i+1 for i in range(len(group[:digit_index[0]]))]
                dpe_digit = [i+2 for i in range(len(group[digit_index[0]:]))]
                dpe_group = dpe_int + dpe_digit
            dpe_group_list.append(((start,end), dpe_group))

        for (start, end), dpe_group in dpe_group_list:
            dpe[start:end+1] = dpe_group

        return dpe   


def categoritize(df, col_name):
    df[col_name] = df[col_name].map(lambda x: stringrize(x, col_name))
    return df


def buckettize(df, code, target_value, quant):
    df[target_value] = df.groupby([code])[target_value].transform(lambda x: x.rank(method = 'dense'))
   
    df[target_value]= df.groupby([code])[target_value].transform(lambda x: q_cut(x,quant))
    return df


def q_cut(x, cuts):

    unique_var = len(np.unique([i for i in x]))
    nunique = len(pd.qcut(x, min(unique_var, cuts), duplicates = 'drop').cat.categories)
    output = pd.qcut(x, min(unique_var,cuts), labels= range(1, min(nunique, cuts)+1), duplicates = 'drop')
    return output


def stringrize(x, col_name):
    if not (x =='nan' or x==pd.isnull(x)):
        return col_name + '_' + str(x)
    else:
        return ' '


def digit_split(digits : str):
    return [' '.join(d) for d in digits]


def isnumeric(text):
    '''returns True if string s is numeric'''    

    return all(s in "0123456789." for s in text) or any(s in "0123456789" for s in text)


def digit_split_in_text(text : str):
    join_list = []
    split = text.split()
    new_split = []
    for text in split:
        if not all(s in "0123456789." for s in text) and any(s in "0123456789" for s in text):
            for i, t in enumerate(text):
                if isnumeric(t):
                    idx = i
            new_split += [text[:idx+1], text[idx+1:]]
        else:
            new_split.append(text)
    split = new_split
   
    for i, d in enumerate(split):
        if isnumeric(d):
            while d.count('.') > 1:
                target = d.rfind('.')
                if target  == (len(d)-1) :
                    d = d[:target]
                else:
                    d = d[:target] + d[(target+1):]
            join_list.append(digit_split(d))

        else:
            join_list.append([d])

    return ' '.join(sum(join_list, []))


def split(word):
    return [char for char in word]


#split and round
def round_digits(digit : str or float or int):
    if isinstance(digit, str):
        return digit_split_in_text(digit)
    elif digit is np.NAN:
        return ' '
    elif isinstance(digit, float):
        return " ".join(split(str(round(digit, 4))))
    elif isinstance(digit, int):
        return " ".join(split(str(digit)))
    elif isinstance(digit, np.int64):
        return str(digit)
    else: 
        return digit


def text_digit_round(text_list):
    if '.' in text_list:
        decimal_point =  text_list.index('.')
        if len(text_list[decimal_point:])> 5:
            return text_list[:decimal_point+5]
        else:
            return text_list
    else:
        return text_list
    
######



######
#
# dataset construct

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

    icu.INTIME, icu.OUTTIME = (
        pd.to_datetime(icu.INTIME),
        pd.to_datetime(icu.OUTTIME)
    )

    icu.sort_values("INTIME", ascending=True, inplace=True)

    icu = icu.groupby("HADM_ID").first()
    icu["HADM_ID"] = icu.index
    icu.reset_index(drop=True, inplace=True)

    cohort = icu
    inputdata_path = os.path.join(args.inputdata_path, "mimiciv_cohort.pkl")
    print(f"The final mimiciv cohort pickle is saved at: {inputdata_path}")
    cohort.to_pickle(inputdata_path)


def main_dataset_construct(args):
    print("create mimiciii ICU start!")
    create_mimiciii_ICU(args)
    print("create eICU ICU start!")
    create_eICU_ICU(args)
    print("create mimiciv ICU start!")
    create_mimiciv_ICU(args)


#####

#####
# main step 1
#
#####
import pandas as pd
import os
from transformers import AutoTokenizer
import argparse

def filter_ID_TIME_NULL(src, config, rawdata_path, inputdata_path, sample=False):

    table_list = []
    column_names = {}

    for table_dict in config["Table"][src]:
        table_name = table_dict["table_name"]
        print("src : ", src, ", table_name : ", table_name)

        df = pd.read_csv(os.path.join(rawdata_path, src, table_name) + ".csv")
        if sample:
            df = df.iloc[: int(len(df) / 1000), :]

        if src == "mimiciv":
            columns_upper(df)
        print("[0] Raw data shape: ", df.shape)

        # 1. Remove columns
        df.drop(columns=table_dict["time_excluded"], inplace=True, errors='ignore')
        df.drop(columns=table_dict["id_excluded"], inplace=True, errors='ignore')
        print("[1] Exclude useless columns: ", df.shape)

        # 2. Rename columns
        df.rename(columns={table_dict["time_column"]: "TIME"}, inplace=True)
        if config["ID"][src] not in df.columns:
            df.rename(columns={config["SUB_ID"][src]: "SUB_ID"}, inplace=True)
        else:
            df.rename(columns={config["ID"][src]: "ID"}, inplace=True)
        print("[2] Rename columns to ID, TIME or SUB_ID, TIME")

        # 3. Map ITEMID into desciprions
        if table_name in config["DICT_FILE"][src].keys():

            dict_name = config["DICT_FILE"][src][table_name][0]
            column_name = config["DICT_FILE"][src][table_name][1]
            dict_path = os.path.join(rawdata_path, src, dict_name + ".csv")
            code_dict = pd.read_csv(dict_path)

            if src == "mimiciv":
                code_dict.columns = map(lambda x: str(x).upper(), code_dict.columns)
            df = name_dict(df, code_dict, column_name)
        print("[3] Map ITEMID into descriptions: ", df.shape)

        # Read ICUSTAY
        icu = pd.read_pickle(os.path.join(inputdata_path, f"{src}_cohort.pkl"))
        if src == "mimiciv":
            columns_upper(icu)
            icu.rename(columns={"ICUSTAY_ID":"STAY_ID"}, inplace=True, errors='ignore')
        
        # 3.5. If there is No coulmns, config["ID"][src]...
        icu.rename(columns={config["ID"][src]: "ID"}, inplace=True)
        if "SUB_ID" in df.columns:
            icu.rename(columns={config["SUB_ID"][src]: "SUB_ID"}, inplace=True)
            df = df.merge(icu[['ID','SUB_ID']], how='left', on='SUB_ID')
            df.drop(columns="SUB_ID", inplace=True, errors='ignore')

        # 4. Filter ID and TIME by ICUSTAY's ID and TIME
        if src in ["mimiciii", "mimiciv"]:
            df = ID_time_filter_mimic(df, icu)
        else:
            df = ID_time_filter_eicu(df, icu)
        if src == "eicu":
            if table_name == "medication":
                df = eicu_med_revise(df)
            elif table_name == "infusionDrug":
                df = eicu_inf_revise(df)
        print("[4] Filter ID,TIME by ICUSTAY ID,TIME: ", df.shape)

        # 5. Filter null columns
        for col in df.columns:
            if df[col].isnull().sum() == len(df):
                df.drop(columns=col, inplace=True)
        print("[5] Filter null columns: ", df.shape)

        # 6. Filter rows where ITEMID == 'nan'
        if table_name in config["DICT_FILE"][src].keys():
            null_itemid_mask = df["ITEMID"] == "nan"
            df = df[~null_itemid_mask]
        print("[6] Filter rows where ITEMID == nan: ", df.shape, "\n")

        # Append
        df["TABLE_NAME"] = table_name
        table_list.append(df)

        column_names[table_name] = list(df.columns)

    # 7. Concat three tables
    cat_df = pd.concat(table_list, axis=0).reset_index(drop=True)

    print(
        "[7] Concat three tables: ",
        cat_df.shape,
        "=",
        table_list[0].shape,
        "+",
        table_list[1].shape,
        "+",
        table_list[2].shape,
        "=",
        (
            table_list[0].shape[0] + table_list[1].shape[0] + table_list[2].shape[0],
            table_list[0].shape[1] + table_list[1].shape[1] + table_list[2].shape[1],
        ),
    )

    # 9. Sort the table
    df_sorted = cat_df.sort_values(["ID", "TIME"], ascending=True)
    print("[9] Sort the concatenated table")

    return df_sorted, column_names


def bucketize_time_gap(df_sorted):

    df_sorted["ORDER"] = list(range(len(df_sorted)))

    # 1. Bucketize time
    df_sorted["time_gap"] = df_sorted.groupby(["ID"])["TIME"].transform(
        lambda x: (x - x.shift(1)).fillna(0)
    )
    df_sorted.reset_index(drop=True, inplace=True)

    df_zero_gap = df_sorted[df_sorted["time_gap"] == 0].reset_index(drop=True)
    df_not_gap = df_sorted[df_sorted["time_gap"] != 0].reset_index(drop=True)

    df_zero_gap["time_bucket"] = "TB_0"
    df_not_gap["time_bucket"] = q_cut(df_not_gap["time_gap"], 20)
    df_not_gap["time_bucket"] = df_not_gap["time_bucket"].apply(
        lambda x: "TB_" + str(x)
    )

    df_time = pd.concat([df_zero_gap, df_not_gap], axis=0).reset_index(drop=True)
    df_time = df_time.sort_values(["ORDER"], ascending=True)

    # 2. Fill null with white space
    df_time.fillna(" ", inplace=True)
    df_time.replace("nan", " ", inplace=True)

    return df_time


def descemb_tokenize(df, table_name):

    target_cols = [
        col
        for col in df.columns
        if col not in ["ID", "TIME", "time_bucket", "time_gap", "TABLE_NAME", "ORDER"]
    ]
    table_token = tokenizer.encode(table_name)[1:-1]

    df[target_cols] = df[target_cols].swifter.applymap(
        lambda x: tokenizer.encode(round_digits(x))[1:-1] if x != " " else []
    )
    df[[col + "_dpe" for col in target_cols]] = df[target_cols].swifter.applymap(
        lambda x: make_dpe(x, number_token_list) if x != [] else []
    )

    df["event"] = df.swifter.apply(
        lambda x: sum(
            [
                tokenizer.encode(col)[1:-1] + x[col]
                for col in target_cols
                if x[col] != []
            ],
            [],
        ),
        axis=1,
    )
    df["type"] = df.swifter.apply(
        lambda x: sum(
            [
                [6] * len(tokenizer.encode(col)[1:-1]) + [7] * len(x[col])
                for col in target_cols
                if x[col] != []
            ],
            [],
        ),
        axis=1,
    )
    df["dpe"] = df.swifter.apply(
        lambda x: sum(
            [
                [1] * len(tokenizer.encode(col)[1:-1]) + x[col + "_dpe"]
                for col in target_cols
                if x[col] != []
            ],
            [],
        ),
        axis=1,
    )

    df["event_token"] = df.apply(
        lambda x: table_token + x["event"] + [vocab[x["time_bucket"]]], axis=1
    )
    df["type_token"] = df.apply(
        lambda x: [5] * len(table_token) + x["type"] + [4], axis=1
    )
    df["dpe_token"] = df.apply(
        lambda x: [1] * len(table_token) + x["dpe"] + [1], axis=1
    )
    return df


def col_select(df, config, src, table_name):
    selected_cols = list(config["selected"][src][table_name].keys()) + [
        "ID",
        "TIME",
        "time_bucket",
        "ORDER",
    ]
    drop_cols = [col for col in list(df.columns) if col not in selected_cols]
    df_drop = df.drop(columns=drop_cols)
    return df_drop


def main_step1(args):

    # Args
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    sample = args.sample
    print("sample: ", sample)

    '''
    # Read config and numeric dict
    config_path = "./json/config.json"
    numeric_path = "./json/numeric_dict.json"

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    with open(numeric_path, "r") as numeric_outfile:
        numeric_dict = json.load(numeric_outfile)
    '''
    config = CONFIG
    numeric_dict = NUMERIC_DICT

    for src in ["mimiciii", "eicu", "mimiciv"]:

        os.makedirs(os.path.join(args.inputdata_path, src), exist_ok=True)

        last_file_pref = config["Table"][src][-1]["table_name"]
        table_names = [elem["table_name"] for elem in config["Table"][src]]

        if not os.path.isfile(
            os.path.join(args.inputdata_path, src, f"{last_file_pref}_temp.csv")
        ):

            # 1.Filter ID, TIME, NULL
            df_1st, column_names = filter_ID_TIME_NULL(
                src, config, args.rawdata_path, args.inputdata_path, sample=sample
            )
            df_temp = df_1st.copy()

            print("Buckettize time gap")
            df_time = bucketize_time_gap(df_temp)
            df = df_time.copy()

            # 2.Split cat table into three tables: LAB, PRESCRIPTIONS, INPUTEVENTS
            print("Split cat table into three tables: LAB, PRESCRIPTIONS, INPUTEVENTS")
            three_dfs = {}
            for table_name in column_names.keys():

                part_df = df[df["TABLE_NAME"] == table_name]

                table_columns = column_names[table_name] + ["time_bucket", "ORDER"]
                part_df = part_df[table_columns]
                three_dfs[table_name] = part_df

                part_df.to_csv(
                    os.path.join(args.inputdata_path, src, f"{table_name}_temp.csv")
                )
                print(table_name, list(part_df.columns))
        else:
            print("Pass 1st & 2nd starges")
            three_dfs = {}
            for table_name in table_names:
                three_dfs[table_name] = pd.read_csv(
                    os.path.join(args.inputdata_path, src, f"{table_name}_temp.csv"),
                    index_col=0,
                )
        # 3.Embed data
        quant = 20
        print("Tokenize start.")
        print(
            "It might take more than five hours in this step. Grap a cup of coffee..."
        )
        for table_dict in config["Table"][src]:
            for preprocess_type in ["whole"]:
                for embed_type in ["descemb"]:
                    
                    table_name = table_dict["table_name"]
                    print(
                        "src : ",
                        src,
                        "table_name : ",
                        table_name,
                        "embed_type : ",
                        embed_type,
                        "preprocess_type : ",
                        preprocess_type,
                    )

                    df = three_dfs[table_name]

                    if preprocess_type == "select":
                        df = col_select(df, config, src, table_name)

                    if embed_type == "descemb":
                        
                        df = descemb_tokenize(df, table_name)
                        df = df[
                            [
                                "ID",
                                "TIME",
                                "time_bucket",
                                "event_token",
                                "type_token",
                                "dpe_token",
                                "ORDER",
                            ]
                        ]
                        print("(EX) ", tokenizer.decode(df["event_token"].iloc[0]))


                    if not os.path.isdir(
                        os.path.join(
                            args.inputdata_path, src, f"{embed_type}_{preprocess_type}"
                        )
                    ):
                        os.makedirs(
                            os.path.join(
                                args.inputdata_path,
                                src,
                                f"{embed_type}_{preprocess_type}",
                            )
                        )

                    df.to_pickle(
                        os.path.join(
                            args.inputdata_path,
                            src,
                            f"{embed_type}_{preprocess_type}",
                            f"{table_name}.pkl",
                        )
                    )
                    print("save " + src + " " + table_name + " to pkl")

#####

#####
#
# main step2 

import pandas as pd
import os, random, time
import numpy as np
import warnings
import more_itertools as mit
from transformers import AutoTokenizer
import json
import argparse
from typing import List


warnings.filterwarnings("ignore")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
print("main_step2.py started")

def complete_dataframe(src, table_names, config, rawdata_path, inputdata_path):

    lab = pd.read_pickle(
        os.path.join(inputdata_path, src, "descemb_whole", f"{table_names[0]}.pkl")
    )
    pre = pd.read_pickle(
        os.path.join(inputdata_path, src, "descemb_whole", f"{table_names[1]}.pkl")
    )
    inp = pd.read_pickle(
        os.path.join(inputdata_path, src, "descemb_whole", f"{table_names[2]}.pkl")
    )
    df = pd.concat([inp, lab, pre], axis=0)
    print(
        ">> Concat inputevents, lab, prescriptions: ",
        df.shape,
        inp.shape[0] + lab.shape[0] + pre.shape[0],
    )

    sorted_df = df.sort_values(["ORDER"], ascending=True).reset_index(drop=True)
    df_agg = sorted_df.groupby(["ID"]).agg(lambda x: x.tolist())
    print(">> Sort by time and liniearize events for each ID")

    # !IMPORTANT
    # Label check
    label_df = pd.read_csv(os.path.join(rawdata_path, "labels", f"{src}_labels.csv"))
    print(f">> loading {src}_labels.csv : {label_df.info()}")
    if src == "mimiciv":
        label_df.rename(columns={config["ID"][src].lower(): "ID"}, inplace=True)
    else:
        label_df.rename(columns={config["ID"][src]: "ID"}, inplace=True)
    print(">> Read cohort dataframe to generate labels")

    # Merge concat dataframe & cohort dataframe
    label_df = pd.merge(label_df, df_agg, how="left", on="ID")
    label_df = label_df.fillna('[]')
    label_df.rename(columns={"ID":"pid"}, inplace=True)

    label_df.to_csv(os.path.join(inputdata_path, src, "label", "label_df.csv"))

    return label_df

def label2target(label: List):
    assert len(label) == 28
    levels = [1 for _ in range(22)]
    levels += [6,6,5,5,5,3]
    
    target = []
    for i in range(len(label)):
        if label[i] == -1:
            target += [0.5 for _ in range(levels[i])]
        else:
            if levels[i] == 1:
                target += [label[i]]
            else:
                tmp = [0 for _ in range(levels[i])]
                tmp[label[i]] = 1
                target += tmp

    assert len(target) == 52
    return target

def split_traintest(label_df, check):

    # One-hot multi-label for diagnosis label
    label_df["labels"] = label_df["labels"].apply(lambda x: label2target(eval(x))) # !IMPORTANT FIX it
    
    # Make random seeds
    seed_list = [2020, 2021, 2022, 2023, 2024]
    test_split = 5
    val_split = 4

    df = label_df.copy()
    for seed in seed_list:
        print("seed split start : ", seed)
        df = stratified_split(df, seed, test_split, val_split)
        df = random_split(df, seed, test_split, val_split)
    
    # Check
    if check:
        # for stratified split
        print("stratified split sanity check !\n")
        col = "labels"
        for seed in seed_list:
            if col in df.columns:
                print(
                    "train = 1/ valid =2 / test= 0 / remove= -1 \n",
                    df[col + f"_{seed}_strat"].value_counts(),
                )
                print(
                    "train label ratio \n ",
                    df.groupby([col, col + f"_{seed}_strat"]).size()
                    / len(df)
                    * 100,
                )

        # for random split
        print("random split sanity check !\n")
        print(
            "train = 1/ valid =2 / test= 0 / remove= -1 \n",
            df[f"{seed}_rand"].value_counts(),
        )
        col = "labels"
        if col in df.columns:
            print(
                "train label ratio \n ",
                df.groupby([col, f"{seed}_rand"]).size() / len(df) * 100,
            )

    return df

def dataframe2numpy(df, root, src):

    np.save(os.path.join(root, src, "npy", "inputs.npy"), df.event_token.to_numpy())
    np.save(os.path.join(root, src, "npy", "types.npy"), df.type_token.to_numpy())
    np.save(os.path.join(root, src, "npy", "dpes.npy"), df.dpe_token.to_numpy())

    np.save(os.path.join(root, src, "label", f"labels.npy"), df['labels'].to_numpy())

def count_events_per_8192(types, max_len):

    event_count = []

    for type_ids in types:

        # Flatten data
        if type(type_ids) == str:
            type_ids = eval(type_ids)
        type_ids = sum(type_ids, [])
        # The first token should be [CLS]
        type_ids.insert(0, 1)

        # Truncate
        if len(type_ids) > max_len - 1:
            flatten_types = np.array(type_ids[:max_len]).astype(np.int32)

            timetoken_mask = flatten_types == 4
            timetoken_idx = np.where(timetoken_mask)[0]
            split_idx = timetoken_idx.max()

            if split_idx == max_len - 1:
                split_idx = sorted(timetoken_idx)[-2]

            # The last token shold be [SEP]
            flatten_types[split_idx + 1] = 2

            # Fill blank with [PAD]
            flatten_types[split_idx + 2 :] = 0

        # Pad
        else:
            flatten_types = np.zeros(max_len).astype(np.int32)
            flatten_types[: len(type_ids)] = type_ids

            # The last token shold be [SEP]
            flatten_types[len(type_ids)] = 2

        # Append
        event_num = (flatten_types == 4).sum()
        event_count.append(event_num)

    return event_count

def match_eventlen(event, event_len, event_max_len, max_len):

    # 1. Match numbers between flat data and hierarchical data
    hi_event = event[:event_len]

    # 2. If the number of events exceeds the maximum, the event is discarded
    if len(hi_event) > event_max_len:
        hi_event = hi_event[:event_max_len]
        return hi_event
    else:
        return hi_event

def make_hi_data(inputs, types, dpes, event_count, event_max_len, word_max_len):

    sample_num = len(event_count)

    input_events_list = np.empty((sample_num, event_max_len, word_max_len)).astype(
        np.int16
    )
    type_events_list = np.empty((sample_num, event_max_len, word_max_len)).astype(
        np.int16
    )
    dpe_events_list = np.empty((sample_num, event_max_len, word_max_len)).astype(
        np.int16
    )
    iter = 0

    events_exceeded = 0

    for input_event, type_event, dpe_event, event_len in zip(
        inputs, types, dpes, event_count
    ):
        # 하.... list로 처리해야하는거 string으로 처리함....
        if type(input_event) == str:
            input_event = eval(input_event)
        if type(type_event) == str:
            type_event = eval(type_event)
        if type(dpe_event) == str:
            dpe_event = eval(dpe_event)

        hi_input_event = match_eventlen(
            input_event, event_len, event_max_len, word_max_len
        )
        hi_type_event = match_eventlen(
            type_event, event_len, event_max_len, word_max_len
        )
        hi_dpe_event = match_eventlen(
            dpe_event, event_len, event_max_len, word_max_len)

        if hi_input_event is None:
            events_exceeded += 1
            continue

        input_words_list = []
        type_words_list = []
        dpe_words_list = []

        for input_ids, type_ids, dpe_ids in zip(
            hi_input_event, hi_type_event, hi_dpe_event
        ):

            # The first token should be [CLS]
            input_ids.insert(0, 101)
            type_ids.insert(0, 1)
            dpe_ids.insert(0, 0)

            if len(type_ids) > word_max_len - 1:

                time_token = input_ids[-1]
                hi_input_ids = np.array(input_ids[:word_max_len]).astype(np.int16)
                hi_type_ids = np.array(type_ids[:word_max_len]).astype(np.int16)
                hi_dpe_ids = np.array(dpe_ids[:word_max_len]).astype(np.int16)

                content_mask = np.where(hi_type_ids == 7)[0]
                content_indices = [
                    list(group) for group in mit.consecutive_groups(content_mask)
                ]
                # content_indices = [[100,101],[107,108]]

                if len(content_indices) == 0:
                    raise AssertionError("Edge case.. Need to check")

                split_indices = content_indices[-1][-1]

                if split_indices >= word_max_len - 2:
                    split_indices = content_indices[-2][-1]

                # The second last token shold be [TIME]
                hi_input_ids[split_indices + 1] = time_token
                hi_type_ids[split_indices + 1] = 4
                hi_dpe_ids[split_indices + 1] = 1

                # The last token shold be [SEP]
                hi_input_ids[split_indices + 2] = 102
                hi_type_ids[split_indices + 2] = 2
                hi_dpe_ids[split_indices + 2] = 0

                # Fill blank with [PAD]
                hi_input_ids[split_indices + 3 :] = 0
                hi_type_ids[split_indices + 3 :] = 0
                hi_dpe_ids[split_indices + 3 :] = 0

            else:
                hi_input_ids = np.zeros(word_max_len).astype(np.int16)
                hi_type_ids = np.zeros(word_max_len).astype(np.int16)
                hi_dpe_ids = np.zeros(word_max_len).astype(np.int16)

                hi_input_ids[: len(input_ids)] = input_ids
                hi_type_ids[: len(type_ids)] = type_ids
                hi_dpe_ids[: len(dpe_ids)] = dpe_ids

                # The last token shold be [SEP]
                hi_input_ids[len(input_ids)] = 102
                hi_type_ids[len(type_ids)] = 2
                hi_dpe_ids[len(dpe_ids)] = 0

            input_words_list.append(hi_input_ids)
            type_words_list.append(hi_type_ids)
            dpe_words_list.append(hi_dpe_ids)

        # Pad events to be 256 events
        empty_event = [
            [0] * word_max_len for i in range(event_max_len - len(input_words_list))
        ]
        input_words_list += empty_event
        type_words_list += empty_event
        dpe_words_list += empty_event

        input_events_list[iter] = input_words_list
        type_events_list[iter] = type_words_list
        dpe_events_list[iter] = dpe_words_list
        iter += 1

    print(f"{events_exceeded} events are discarded..")
    if iter != sample_num:
        raise AssertionError("Should be the same!")

    return {
        "inputs": input_events_list,
        "types": type_events_list,
        "dpes": dpe_events_list,
    }

def flatten_hi_data(hi_output, max_len):

    input_ids_list = []
    type_ids_list = []
    dpe_ids_list = []
    event_count = []

    for input_ids, type_ids, dpe_ids in zip(
        hi_output["inputs"], hi_output["types"], hi_output["dpes"]
    ):

        # Flatten data
        input_ids = input_ids.flatten()
        type_ids = type_ids.flatten()
        dpe_ids = dpe_ids.flatten()

        # Remove [PAD], [CLS], [SEQ]
        mask = (input_ids == 0) | (input_ids == 101) | (input_ids == 102)

        input_ids = input_ids[~mask]
        type_ids = type_ids[~mask]
        dpe_ids = dpe_ids[~mask]

        padded_input_ids = np.zeros(max_len).astype(np.int16)
        padded_type_ids = np.zeros(max_len).astype(np.int16)
        padded_dpe_ids = np.zeros(max_len).astype(np.int16)

        # The first token shold be [CLS]
        padded_input_ids[0] = 101
        padded_type_ids[0] = 1
        padded_dpe_ids[0] = 0

        padded_input_ids[1 : len(input_ids) + 1] = input_ids
        padded_type_ids[1 : len(type_ids) + 1] = type_ids
        padded_dpe_ids[1 : len(dpe_ids) + 1] = dpe_ids

        # The last token shold be [SEP]
        padded_input_ids[len(input_ids) + 1] = 102
        padded_type_ids[len(type_ids) + 1] = 2
        padded_dpe_ids[len(dpe_ids) + 1] = 0

        input_ids_list.append(padded_input_ids)
        type_ids_list.append(padded_type_ids)
        dpe_ids_list.append(padded_dpe_ids)

        event_num = (padded_type_ids == 4).sum()
        event_count.append(event_num)

    return {
        "inputs": np.array(input_ids_list),
        "types": np.array(type_ids_list),
        "dpes": np.array(dpe_ids_list),
        "event_count": event_count,
    }

def final_check(flat_output, hi_output, word_max_len):

    print("FLAT OUTPUT CHECK")
    p_idx = random.randint(0, len(flat_output))

    for ii, ti, di in zip(
        flat_output["inputs"][p_idx][:128],
        flat_output["types"][p_idx][:128],
        flat_output["dpes"][p_idx][:128],
    ):
        print(ti, "\t", di, "\t", tokenizer.decode(ii))

    print("\n...\n")

    for ii, ti, di in zip(
        flat_output["inputs"][p_idx][-128:],
        flat_output["types"][p_idx][-128:],
        flat_output["dpes"][p_idx][-128:],
    ):
        print(ti, "\t", di, "\t", tokenizer.decode(ii))

    flat_event_num = (flat_output["types"][p_idx] == 4).sum()

    print("\nHIERARHCICAL OUTPUT CHECK")

    hi_event_num = 0
    for input_ids in hi_output["inputs"][p_idx]:
        if (input_ids == 0).sum() != word_max_len:
            hi_event_num += 1

    w_idx = 0
    for ii, ti, di in zip(
        hi_output["inputs"][p_idx][w_idx],
        hi_output["types"][p_idx][w_idx],
        hi_output["dpes"][p_idx][w_idx],
    ):
        print(ti, "\t", di, "\t", tokenizer.decode(ii))

    if flat_event_num != hi_event_num:
        raise AssertionError("Should be the same")
    return

def main_step2(args):
    # Argument
    save = True
    config = CONFIG

    for src in ["mimiciii", "eicu", "mimiciv"]:
        if os.path.isfile(os.path.join(args.inputdata_path, src, "fold", "fold_5.csv")):
            continue
        
        os.makedirs(os.path.join(args.inputdata_path, src), exist_ok=True)
        os.makedirs(os.path.join(args.inputdata_path, src, "label"), exist_ok=True)
        os.makedirs(os.path.join(args.inputdata_path, src, "fold"), exist_ok=True)
        os.makedirs(os.path.join(args.inputdata_path, src, "npy"), exist_ok=True)

        table_names = [elem["table_name"] for elem in config["Table"][src]]

        # [1] Find label and make fold splits
        if not os.path.isfile(os.path.join(args.inputdata_path, src, "label", "label_df.csv")): # 아니 진짜 오타 개많네
            df = complete_dataframe(src, table_names, config, args.rawdata_path, args.inputdata_path)
        else:
            start = time.time()
            df = pd.read_csv(os.path.join(args.inputdata_path, src, "label", "label_df.csv"))
            end = time.time()
            print(">> Loading {} csv is done... {} [sec]".format(df.shape, end - start))

        if not os.path.isfile(os.path.join(args.inputdata_path, src, "label", "labels.npy")):
            # [2] Convert df to numpy
            dataframe2numpy(df, args.inputdata_path, src)

        start = time.time()
        inputs = np.load(
            os.path.join(args.inputdata_path, src, "npy", "inputs.npy"), allow_pickle=True
        )
        types = np.load(
            os.path.join(args.inputdata_path, src, "npy", "types.npy"), allow_pickle=True
        )
        dpes = np.load(
            os.path.join(args.inputdata_path, src, "npy", "dpes.npy"), allow_pickle=True
        )
        end = time.time()
        print(
            ">> Loading numpy {} is done... {} [sec]".format(inputs.shape, end - start)
        )
        
        # [3] Fix-length data
        event_count = count_events_per_8192(types, max_len=8192)

        start = time.time()
        hi_output = make_hi_data(
            inputs, types, dpes, event_count, event_max_len=256, word_max_len=128
        )
        print(
            ">> Hierarhical data {} is done... {} [sec]".format(
                hi_output["inputs"].shape, time.time() - start
            )
        )

        start = time.time()
        flat_output = flatten_hi_data(hi_output, max_len=8192)
        print(
            ">> Flat data {} is done... {} [sec]".format(
                flat_output["inputs"].shape, time.time() - start
            )
        )

        # mask = np.array(event_count) > 256
        # print(f">> {mask.sum()} events are discared ..")
        if save:
            np.save(
                os.path.join(args.inputdata_path, src, "npy", "inputs_ids.npy"),
                hi_output["inputs"],
            )
            np.save(
                os.path.join(args.inputdata_path, src, "npy", "type_ids.npy"),
                hi_output["types"],
            )
            np.save(
                os.path.join(args.inputdata_path, src, "npy", "dpe_ids.npy"),
                hi_output["dpes"],
            )
            
            split_traintest(df.reset_index(drop=False), check=False).to_csv(
                os.path.join(args.inputdata_path, src, "fold", f"fold_5.csv")
            )


#####
#
# npy_script


def main_npy_script(args):

    srcs = ["mimiciii", "eicu", "mimiciv"]

    for src in srcs:
        input_path = os.path.join(args.inputdata_path, src, "npy")
        inputs = np.load(
            os.path.join(input_path, f"inputs_ids.npy"),
            allow_pickle=True,
        ).astype(np.int64)

        types = np.load(
            os.path.join(input_path, f"type_ids.npy"),
            allow_pickle=True,
        ).astype(np.int64)

        dpes = np.load(
            os.path.join(input_path, f"dpe_ids.npy"),
            allow_pickle=True,
        ).astype(np.int64)

        if not os.path.isdir( os.path.join(input_path, "dpe_ids") ):
            os.mkdir(os.path.join(input_path, "inputs_ids"))
            os.mkdir(os.path.join(input_path, "type_ids"))
            os.mkdir(os.path.join(input_path, "dpe_ids"))

        for idx, inp in enumerate(inputs):
            remove_zero_events = np.array([i for i in inp if np.any(i)])
            trunc_inp = np.array([(i[i != 0]).tolist() for i in remove_zero_events])
            len_events = len(remove_zero_events)
            len_tokens = [len(i) for i in trunc_inp]

            trunc_type = [
                i[: len_tokens[idx_i]].tolist()
                for idx_i, i in enumerate(types[idx][:len_events])
            ]
            trunc_dpes = [
                i[: len_tokens[idx_i]].tolist()
                for idx_i, i in enumerate(dpes[idx][:len_events])
            ]
            # breakpoint()

            
            np.save(
                os.path.join(input_path, "inputs_ids", f"{idx}.npy"),
                trunc_inp,
            )
            np.save(
                os.path.join(input_path, "type_ids", f"{idx}.npy"),
                trunc_type,
            )
            np.save(
                os.path.join(input_path, "dpe_ids", f"{idx}.npy"),
                trunc_dpes,
            )
        print(f"[{src}] Finished creating npy datasets for {src}...")


#####

def get_parser():
    """
    Note:
        Do not add command-line arguments here when you submit the codes.
        Keep in mind that we will run your pre-processing code by this command:
        `python 00000000_preprocess.py ./train --dest ./output`
        which means that we might not be able to control the additional arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        metavar="DIR",
        help="root directory containing different ehr files to pre-process (usually, 'train/')"
    )
    parser.add_argument(
        "--dest",
        type=str,
        metavar="DIR",
        help="output directory"
    )
    parser.add_argument(
        "--sample_filtering",
        action='store_true',
        help="indicator to prevent filtering from being applies to the test dataset."
    )
    return parser

def main(args):
    """
    TODO:
        Implement your feature preprocessing function here.
        Rename the file name with your student number.
    
    Note:
        1. This script should dump processed features to the --dest directory.
        Note that --dest directory will be an input to your dataset class (i.e., --data_path).
        You can dump any type of files such as json, cPickle, or whatever your dataset can handle.

        2. If you use vocabulary, you should specify your vocabulary file(.pkl) in this code section.
        Also, you must submit your vocabulary file({student_id}_vocab.pkl) along with the scripts.
        Example:
            with open('./20231234_vocab.pkl', 'rb') as f:
                (...)

        3. For fair comparison, we do not allow to filter specific samples when using test dataset.
        Therefore, if you filter some samples from the train dataset,
        you must use the '--sample_filtering' argument to prevent filtering from being applied to the test dataset.
        We will set the '--sample_filtering' argument to False and run the code for inference.
        We also check the total number of test dataset.
    """

    root_dir = args.root
    dest_dir = args.dest
    os.makedirs(dest_dir, exist_ok=True)

    args = my_args(root_dir, dest_dir)
    
    # dataset_construct
    main_dataset_construct(args)

    # main step 1
    main_step1(args)

    # main step 2
    main_step2(args)

    # npy script
    main_npy_script(args)

    


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)