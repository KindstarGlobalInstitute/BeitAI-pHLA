import math
import os

import numpy as np
import pandas as pd
import xlrd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

Result_path = "./output_MIL/"
best_th_TripHLApan = 0.9142618417739868 
best_th_CapsNet = 0.9773489356040954 
best_th_BeitAI = 0.5876

def calculate_best_threshold(actual, prob):
    precisions, recalls, thresholds = precision_recall_curve(actual, prob)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)

    if len(f1_scores) == 0:
        best_threshold = 0.5
    else:
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
    return best_threshold

def calculate_metrics(actual, prob, pred):
    auroc = roc_auc_score(actual, prob)
    auprc = average_precision_score(actual, prob)
    ppv = precision_score(actual, pred)
    f1 = f1_score(actual, pred)
    return auroc, auprc, ppv, f1

def merge_data_SA():
    BeitAI_res_path = f"{Result_path}SA_test_seen/BeitAI-pHLA_result/"
    NetMHCpan_path = f"{Result_path}SA_test_seen/NetMHCpan_result_static/"
    MHCflurry_path = f"{Result_path}SA_test_seen/MHCflurry_result/"

    df_all = []
    for file in os.listdir(BeitAI_res_path):
        mhc_name = file.replace(".txt", "")
        NetMHCpan_df = pd.read_csv(NetMHCpan_path + file, sep="\t", header=None)
        MHCflurry_df = pd.read_csv(MHCflurry_path + file, sep="\t", header=None)
        BeitAI_df = pd.read_csv(BeitAI_res_path + file, sep="\t", header=None)

        NetMHCpan_df.columns = ["epitope", "NetMHCpan", "NetMHCpan_pred"]
        MHCflurry_df.columns = ["epitope", "MHCflurry", "MHCflurry_pred", "MHCflurry_prob"]

        BeitAI_df.columns = ["epitope", "label", "mhc", "pred_0", "pred_1", "pred_2"]
        BeitAI_df["BeitAI-pHLA"] = BeitAI_df[["pred_0", "pred_1", "pred_2"]].mean(axis=1)

        merged = BeitAI_df  # 初始
        merged = pd.merge(merged, NetMHCpan_df, on='epitope', how='outer')
        merged = pd.merge(merged, MHCflurry_df, on='epitope', how='outer')

        merged['MHCflurry_pred'] = (merged['MHCflurry'] < 0.5).astype(int)
        merged['NetMHCpan_pred'] = (merged['NetMHCpan'] < 0.5).astype(int)
        merged['BeitAI-pHLA_pred'] = (merged['BeitAI-pHLA'] >= best_th_BeitAI).astype(int)
        merged['MHC_name'] = mhc_name

        merged = merged[['MHC_name', "epitope", "label", "NetMHCpan", "MHCflurry", "BeitAI-pHLA", "NetMHCpan_pred", 'MHCflurry_pred', "BeitAI-pHLA_pred"]]
        cols_to_convert = ['label', "NetMHCpan_pred", "BeitAI-pHLA_pred"]
        merged[cols_to_convert] = merged[cols_to_convert].astype(int)
        df_all.append(merged)
    merged_all_df = pd.concat(df_all, ignore_index=True)
    merged_all_df.to_csv(f"{Result_path}SA_test_seen/SA_data_result.csv", index=False, encoding='utf-8')


def merge_data_MA():
    BeitAI_res_path = f"{Result_path}MA_test/BeitAI-pHLA_result/"
    NetMHCpan_path = f"{Result_path}MA_test/NetMHCpan_result_static/"
    MHCflurry_path = f"{Result_path}MA_test/MHCflurry_result/"
    MixMHCpred_path = f"{Result_path}MA_test/MixMHCpred_result_2.2_static/"
    TripHLApan_path = f"{Result_path}MA_test/TripHLApan_result_static/"
    CapsNet_path = f"{Result_path}MA_test/CapsNet_result/"


    df_all = []
    for file in os.listdir(NetMHCpan_path):
        mhc_name = file.replace(".txt", "")
        NetMHCpan_df = pd.read_csv(NetMHCpan_path + file, sep="\t", header=None)
        MixMHCpred_df = pd.read_csv(MixMHCpred_path + file, sep="\t", header=None)
        MHCflurry_df = pd.read_csv(MHCflurry_path + file, sep="\t", header=None)
        TripHLApan_df = pd.read_csv(TripHLApan_path + file, sep=",", header=0)
        CapsNet_df = pd.read_csv(CapsNet_path + file, sep="\t", header=0)
        BeitAI_df = pd.read_csv(BeitAI_res_path + file, sep="\t", header=None)


        MixMHCpred_df.columns = ["epitope", "MixMHCpred"]
        NetMHCpan_df.columns = ["epitope", "NetMHCpan", "NetMHCpan_pred"]
        MHCflurry_df.columns = ["epitope", "MHCflurry", "MHCflurry_pred", "MHCflurry_prob"]
        TripHLApan_df.columns = ["epitope", "TripHLApan"]
        CapsNet_df.columns = ["epitope", "label", "CapsNet-MHC"]
        CapsNet_df = CapsNet_df.drop(columns=["label"])

        BeitAI_df.columns = ["epitope", "label", "pred_0", "pred_1", "pred_2"]
        BeitAI_df["BeitAI-pHLA"] = BeitAI_df[["pred_0", "pred_1", "pred_2"]].mean(axis=1)

        merged = BeitAI_df
        merged = pd.merge(merged, MixMHCpred_df, on='epitope', how='outer')
        merged = pd.merge(merged, NetMHCpan_df, on='epitope', how='outer')
        merged = pd.merge(merged, MHCflurry_df, on='epitope', how='outer')
        merged = pd.merge(merged, TripHLApan_df, on='epitope', how='outer')
        merged = pd.merge(merged, CapsNet_df, on='epitope', how='outer')


        merged['MHC_name'] = mhc_name

        # merged = merged[['MHC_name', "epitope", "label", "MixMHCpred", "NetMHCpan", "MHCflurry", "TripHLApan", "CapsNet-MHC", "BeitAI-pHLA",
        #                  'MixMHCpred_pred', "NetMHCpan_pred", 'MHCflurry_pred', 'TripHLApan_pred', "CapsNet-MHC_pred", "BeitAI-pHLA_pred"]]
        merged = merged[['MHC_name', "epitope", "label", "MixMHCpred", "NetMHCpan", "MHCflurry", "TripHLApan", "CapsNet-MHC", "BeitAI-pHLA"]]

        # cols_to_convert = ['label', "NetMHCpan_pred", "CapsNet-MHC_pred", "BeitAI-pHLA_pred"]
        # merged[cols_to_convert] = merged[cols_to_convert].astype(int)
        df_all.append(merged)
    merged_all_df = pd.concat(df_all, ignore_index=True)
    merged_all_df.to_csv(f"{Result_path}MA_test/MA_data_result.csv", index=False, encoding='utf-8')

def merge_data_MixMHCpred():

    Result_path = "./output_MIL/MixMHCpred_test/"
    file_path = "./output_MIL/MixMHCpred_test/TableS3.xlsx"
    our_res_path = "./output_MIL/MixMHCpred_test/BeitAI-pHLA_result/"
    TripHLApan_res_path = './output_MIL/MixMHCpred_test/MA_MixMHCpred_result/'
    CapsNet_res_path = f"{Result_path}CapsNet_result/"
    with pd.ExcelFile(file_path) as xls:
        dfs = []
        for sheet_name in xls.sheet_names:
            data = xls.parse(sheet_name, skiprows=1, engine='openpyxl')

            # 添加BeitAI-pHLA结果
            our_res_df = pd.read_csv(our_res_path + sheet_name + ".txt", sep="\t", header=None)
            our_res_df.columns = ["epitope", "label", "mhc", "pred_0", "pred_1", "pred_2"]
            our_res_df["BeitAI-pHLA"] = our_res_df[["pred_0", "pred_1", "pred_2"]].mean(axis=1)
            our_res_df = our_res_df[["epitope", "BeitAI-pHLA"]]
            if data.shape[0] == our_res_df.shape[0]:
                data = pd.merge(data, our_res_df, left_on='Sequence', right_on='epitope', how="left")
            else:
                print(sheet_name)
            data = data.drop(columns=['epitope'])

            # 添加TripHLApan结果
            TripHLApan_res_df = pd.read_csv(TripHLApan_res_path + sheet_name + ".csv", header=None)
            df_prob = TripHLApan_res_df.iloc[:, 1:-1]
            pred_scores = df_prob.max(axis=1)

            TripHLApan_res_df['pred'] = pred_scores

            TripHLApan_res_df = TripHLApan_res_df[[0, 'pred']]
            TripHLApan_res_df.columns = ["epitope", "TripHLApan"]

            if data.shape[0] == TripHLApan_res_df.shape[0]:
                data = pd.merge(data, TripHLApan_res_df, left_on='Sequence', right_on='epitope', how="left")
            else:
                print(sheet_name)
            data = data.drop(columns=['epitope'])

            # 添加CapsNet结果
            CapsNet_res_df = pd.read_csv(CapsNet_res_path + sheet_name + ".txt", sep="\t", header=0)
            CapsNet_res_df['CapsNet-MHC'] = CapsNet_res_df['prediction']
            CapsNet_res_df['epitope'] = CapsNet_res_df['peptide']

            CapsNet_res_df_final = CapsNet_res_df[['epitope', 'CapsNet-MHC']]

            if data.shape[0] == CapsNet_res_df_final.shape[0]:
                data = pd.merge(data, CapsNet_res_df_final, left_on='Sequence', right_on='epitope', how="left")
            else:
                print(sheet_name)
            data = data.drop(columns=['epitope'])


            data['sampleName'] = sheet_name
            # sampleName放到第一列
            data = data[['sampleName'] + [col for col in data.columns if col != 'sampleName']]
            dfs.append(data)
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = merged_df[['sampleName', 'Sequence', 'Ligand', 'MixMHCpred2.2', 'NetMHCpan4.1', 'MHCflurry2.0',
                               'TripHLApan', 'CapsNet-MHC', 'BeitAI-pHLA']]
        merged_df.columns = ['sampleName', 'epitope', 'label', 'MixMHCpred2.2', 'NetMHCpan4.1', 'MHCflurry2.0',
                             'TripHLApan', 'CapsNet-MHC', 'BeitAI-pHLA']


        # 将合并后的 DataFrame 输出为 CSV 文件
        merged_df.to_csv(f"{Result_path}MixMHCpred_data_result.csv", index=False, encoding='utf-8')

def calc_MA_auc():
    data = pd.read_csv(f"{Result_path}MA_test/MA_data_result.csv")

    sample_list = data['MHC_name'].unique().tolist()
    MA_AUC_file = open(f"{Result_path}MA_test_AUC", "w")
    MA_AUPRC_file = open(f"{Result_path}MA_test_AUPRC", "w")
    MA_PPV_file = open(f"{Result_path}MA_test_PPV", "w")
    MA_F1_file = open(f"{Result_path}MA_test_F1", "w")

    # # 计算最佳阈值
    # best_th_TripHLApan = calculate_best_threshold(data['label'], data['TripHLApan'])
    # best_th_CapsNet = calculate_best_threshold(data['label'], data['CapsNet-MHC'])
    # best_th_BeitAI = calculate_best_threshold(data['label'], data['BeitAI-pHLA'])

    # print("best threshold TripHLApan: ", best_th_TripHLApan, "\n")
    # print("best threshold CapsNet: ", best_th_CapsNet, "\n")
    # print("best threshold BeitAI: ", best_th_BeitAI, "\n")

    data['MixMHCpred_pred'] = (data['MixMHCpred'] < 0.5).astype(int)
    data['MHCflurry_pred'] = (data['MHCflurry'] < 0.5).astype(int)
    data['NetMHCpan_pred'] = (data['NetMHCpan'] < 0.5).astype(int)
    data['TripHLApan_pred'] = (data['TripHLApan'] >= best_th_TripHLApan).astype(int)
    data['CapsNet-MHC_pred'] = (data['CapsNet-MHC'] >= best_th_CapsNet).astype(int)
    data['BeitAI-pHLA_pred'] = (data['BeitAI-pHLA'] >= best_th_BeitAI).astype(int)

    data['MixMHCpred'] = (100 - data['MixMHCpred']) / 100
    data['NetMHCpan'] = (100 - data['NetMHCpan']) / 100
    data['MHCflurry'] = (100 - data['MHCflurry']) / 100

    for sample in sample_list:
        sample_data = data[data['MHC_name'] == sample]
        df_balanced = sample_data

        auc_Mix2_2, ap_Mix2_2, ppv_Mix2_2, F1_Mix2_2 = calculate_metrics(df_balanced['label'], df_balanced['MixMHCpred'], df_balanced['MixMHCpred_pred'])
        auc_Net, ap_Net, ppv_Net, F1_Net = calculate_metrics(df_balanced['label'], df_balanced['NetMHCpan'], df_balanced['NetMHCpan_pred'])
        auc_flurry, ap_flurry, ppv_flurry, F1_flurry = calculate_metrics(df_balanced['label'], df_balanced['MHCflurry'], df_balanced['MHCflurry_pred'])
        auc_TripHLApan, ap_TripHLApan, ppv_TripHLApan, F1_TripHLApan = calculate_metrics(df_balanced['label'], df_balanced['TripHLApan'], df_balanced['TripHLApan_pred'])
        auc_CapsNet, ap_CapsNet, ppv_CapsNet, F1_CapsNet = calculate_metrics(df_balanced['label'], df_balanced['CapsNet-MHC'], df_balanced['CapsNet-MHC_pred'])
        auc_our, ap_our, ppv_our, F1_our = calculate_metrics(df_balanced['label'], df_balanced['BeitAI-pHLA'], df_balanced['BeitAI-pHLA_pred'])

        MA_AUC_file.write(str(auc_our) + "\t" + str(auc_Mix2_2) + "\t" + str(auc_Net) + "\t" + str(auc_flurry) + "\t" + str(auc_TripHLApan) + "\t" + str(auc_CapsNet) + "\n")
        MA_AUPRC_file.write(str(ap_our) + "\t" + str(ap_Mix2_2) + "\t" + str(ap_Net) + "\t" + str(ap_flurry) + "\t" + str(ap_TripHLApan) + "\t" + str(ap_CapsNet) + "\n")
        MA_PPV_file.write(str(ppv_our) + "\t" + str(ppv_Mix2_2) + "\t" + str(ppv_Net) + "\t" + str(ppv_flurry) + "\t" + str(ppv_TripHLApan) + "\t" + str(ppv_CapsNet) + "\n")
        MA_F1_file.write(str(F1_our) + "\t" + str(F1_Mix2_2) + "\t" + str(F1_Net) + "\t" + str(F1_flurry) + "\t" + str(F1_TripHLApan) + "\t" + str(F1_CapsNet) + "\n")
    MA_AUC_file.close()
    MA_AUC_file.close()
    MA_AUPRC_file.close()
    MA_PPV_file.close()
    MA_F1_file.close()

def calc_SA_auc():
    data = pd.read_csv(f"{Result_path}SA_test_seen/SA_data_result.csv")
    data['NetMHCpan'] = (100 - data['NetMHCpan']) / 100
    data['MHCflurry'] = (100 - data['MHCflurry']) / 100

    sample_list = data['MHC_name'].unique().tolist()
    MA_AUC_file = open(f"{Result_path}SA_test_seen_AUC", "w")
    MA_AUPRC_file = open(f"{Result_path}SA_test_seen_AUPRC", "w")
    MA_PPV_file = open(f"{Result_path}SA_test_seen_PPV", "w")
    MA_F1_file = open(f"{Result_path}SA_test_seen_F1", "w")

    # SA_res_file = open(f"{Result_path}SA_test_unseen_total", "w")

    for sample in sample_list:
        sample_data = data[data['MHC_name'] == sample]
        df_balanced = sample_data

        auc_Net, ap_Net, ppv_Net, F1_Net = calculate_metrics(df_balanced['label'], df_balanced['NetMHCpan'], df_balanced['NetMHCpan_pred'])
        auc_flurry, ap_flurry, ppv_flurry, F1_flurry = calculate_metrics(df_balanced['label'], df_balanced['MHCflurry'], df_balanced['MHCflurry_pred'])
        auc_our, ap_our, ppv_our, F1_our = calculate_metrics(df_balanced['label'], df_balanced['BeitAI-pHLA'], df_balanced['BeitAI-pHLA_pred'])

        MA_AUC_file.write(sample + "\t" + str(auc_our) + "\t" + str(auc_Net) + "\t" + str(auc_flurry) + "\n")
        MA_AUPRC_file.write(sample + "\t" + str(ap_our) + "\t" + str(ap_Net) + "\t" + str(ap_flurry) + "\n")
        MA_PPV_file.write(sample + "\t" + str(ppv_our) + "\t" + str(ppv_Net) + "\t" + str(ppv_flurry) + "\n")
        MA_F1_file.write(sample + "\t" + str(F1_our) + "\t" + str(F1_Net) + "\t" + str(F1_flurry) + "\n")
    MA_AUC_file.close()
    MA_AUPRC_file.close()
    MA_PPV_file.close()
    MA_F1_file.close()


def calc_MixMHCpred_auc():
    data = pd.read_csv(f"{Result_path}MixMHCpred_test/MixMHCpred_data_result.csv")

    data['MixMHCpred_pred'] = (data['MixMHCpred2.2'] < 0.5).astype(int)
    data['NetMHCpan_pred'] = (data['NetMHCpan4.1'] < 0.5).astype(int)
    data['MHCflurry_pred'] = (data['MHCflurry2.0'] < 0.5).astype(int)
    data['TripHLApan_pred'] = (data['TripHLApan'] >= best_th_TripHLApan).astype(int)
    data['CapsNet-MHC_pred'] = (data['CapsNet-MHC'] >= best_th_CapsNet).astype(int)
    data['BeitAI-pHLA_pred'] = (data['BeitAI-pHLA'] >= best_th_BeitAI).astype(int)

    data['MixMHCpred2.2'] = (100 - data['MixMHCpred2.2']) / 100
    data['NetMHCpan4.1'] = (100 - data['NetMHCpan4.1']) / 100
    data['MHCflurry2.0'] = (100 - data['MHCflurry2.0']) / 100

    sample_list = data['sampleName'].unique().tolist()
    MA_AUC_file = open(f"{Result_path}MixMHCpred_test_AUC", "w")
    MA_AUPRC_file = open(f"{Result_path}MixMHCpred_test_AUPRC", "w")
    MA_PPV_file = open(f"{Result_path}MixMHCpred_test_PPV", "w")
    MA_F1_file = open(f"{Result_path}MixMHCpred_test_F1", "w")

    for sample in sample_list:
        sample_data = data[data['sampleName'] == sample]
        df_balanced = sample_data

        auc_Mix2_2, ap_Mix2_2, ppv_Mix2_2, F1_Mix2_2 = calculate_metrics(df_balanced['label'], df_balanced['MixMHCpred2.2'], df_balanced['MixMHCpred_pred'])
        auc_Net, ap_Net, ppv_Net, F1_Net = calculate_metrics(df_balanced['label'], df_balanced['NetMHCpan4.1'], df_balanced['NetMHCpan_pred'])
        auc_flurry, ap_flurry, ppv_flurry, F1_flurry = calculate_metrics(df_balanced['label'], df_balanced['MHCflurry2.0'], df_balanced['MHCflurry_pred'])
        auc_TripHLApan, ap_TripHLApan, ppv_TripHLApan, F1_TripHLApan = calculate_metrics(df_balanced['label'], df_balanced['TripHLApan'], df_balanced['TripHLApan_pred'])
        auc_CapsNet, ap_CapsNet, ppv_CapsNet, F1_CapsNet = calculate_metrics(df_balanced['label'], df_balanced['CapsNet-MHC'], df_balanced['CapsNet-MHC_pred'])
        auc_our, ap_our, ppv_our, F1_our = calculate_metrics(df_balanced['label'], df_balanced['BeitAI-pHLA'], df_balanced['BeitAI-pHLA_pred'])

        MA_AUC_file.write(str(auc_our) + "\t" + str(auc_Mix2_2) + "\t" + str(auc_Net) + "\t" + str(auc_flurry) + "\t" + str(auc_TripHLApan) + "\t" + str(auc_CapsNet) + "\n")
        MA_AUPRC_file.write(str(ap_our) + "\t" + str(ap_Mix2_2) + "\t" + str(ap_Net) + "\t" + str(ap_flurry) + "\t" + str(ap_TripHLApan) + "\t" + str(ap_CapsNet) + "\n")
        MA_PPV_file.write(str(ppv_our) + "\t" + str(ppv_Mix2_2) + "\t" + str(ppv_Net) + "\t" + str(ppv_flurry) + "\t" + str(ppv_TripHLApan) + "\t" + str(ppv_CapsNet) + "\n")
        MA_F1_file.write(str(F1_our) + "\t" + str(F1_Mix2_2) + "\t" + str(F1_Net) + "\t" + str(F1_flurry) + "\t" + str(F1_TripHLApan) + "\t" + str(F1_CapsNet) + "\n")
    MA_AUC_file.close()
    MA_AUPRC_file.close()
    MA_PPV_file.close()
    MA_F1_file.close()



def _best_hla(row, cols, hla_list):
    lists = [list(map(float, row[col].split(','))) for col in cols]
    avg_list = np.mean(lists, axis=0)

    sorted_indices = np.argsort(avg_list)
    max_idx = sorted_indices[-1]
    second_max_idx = sorted_indices[-2]

    max_val = avg_list[max_idx]
    second_max_val = avg_list[second_max_idx]
    diff = max_val - second_max_val
    # return diff, hla_list
    return diff, hla_list[max_idx], hla_list[second_max_idx]

def calc_BeitAI_attention_distribution():
    # 加载HLA类型
    allele_HLA_dict = {}
    with open("./data_random/HLA_test.txt", 'r') as f:
        for line in f:
            line = line.strip()
            name, allele = line.split('\t', maxsplit=1)
            allele_list = allele.split("\t")
            allele_HLA_dict[name] = allele_list

    # 加载超型
    allele_supertype = {}
    with open("./data_random/Allele_Supertype.txt", 'r') as f:
        next(f)
        for line in f:
            line = line.strip()
            name, supertype = line.split('\t', maxsplit=1)
            allele = name.replace("*", "")
            allele = "HLA-" + allele[:-2] + ":" + allele[-2:]
            allele_supertype[allele] = supertype

    Result_path = "./output_MIL/MixMHCpred_test/BeitAI-pHLA_result/attention_result/"
    res_list = []
    res_df = pd.DataFrame()
    for file in os.listdir(Result_path):
        allele_name = file.replace(".txt", "")
        HLA_list = allele_HLA_dict[allele_name]
        f_df = pd.read_csv(f"{Result_path}{file}", sep="\t", index_col=False, header=None, names=["pep", "label", "prob_0", "att_list_0", "prob_1", "att_list_1", "prob_2", "att_list_2"])
        f_df['prob'] = f_df[['prob_0', 'prob_1', 'prob_2']].mean(axis=1)
        f_df['pred'] = f_df.apply(lambda row: 1 if row['prob'] >= 0.5 else 0, axis=1)
        f_df['allele'] = allele_name
        f_df_filter = f_df[f_df["pred"] == 1]
        # f_df_filter['best_hla'] = f_df_filter['att_list'].apply(lambda s: HLA_list[np.argmax(list(map(float, s.split(','))))])
        f_df_filter['difference'] = f_df_filter.apply(lambda row: _best_hla(row, ["att_list_0", "att_list_1", "att_list_2"], HLA_list)[0],axis=1)
        f_df_filter['top_allele'] = f_df_filter.apply(lambda row: _best_hla(row, ["att_list_0", "att_list_1", "att_list_2"], HLA_list)[1],axis=1)
        f_df_filter['sec_allele'] = f_df_filter.apply(lambda row: _best_hla(row, ["att_list_0", "att_list_1", "att_list_2"], HLA_list)[2],axis=1)
        res_list += f_df_filter['difference'].tolist()
        res_df_tmp = f_df_filter[['allele', 'pep', 'difference', 'top_allele', 'sec_allele']]
        res_df = pd.concat([res_df, res_df_tmp], ignore_index=True)

        for hla in HLA_list:
            if hla.startswith("HLA-C"):
                allele_supertype[hla] = "B7"
            elif hla not in allele_supertype.keys():
                allele_supertype[hla] = "Unclassified"

    # res_df['top_allele'] = res_df['top_allele'].map(lambda x: allele_supertype[x])
    # res_df['sec_allele'] = res_df['sec_allele'].map(lambda x: allele_supertype[x])
    # res_df['is_supertype'] = res_df.apply(lambda row: "1" if (not set(row['top_allele'].split(' ')).isdisjoint(set(row['sec_allele'].split(' ')))) and row['top_allele'] != "Unclassified" and row['sec_allele'] != "Unclassified" else "0" ,axis=1)
    # res_df.to_csv("./output_MIL/MixMHCpred_test/BeitAI-pHLA_result/attention_dist.csv")
    bins = np.arange(0, 1.05, 0.05)
    # plt.hist(res_list, bins=bins, density=True, edgecolor='black', alpha=0.7)
    counts, bins = np.histogram(res_list, bins=bins)
    freq = counts / len(res_list)   # 百分比
    plt.bar(bins[:-1], freq, width=0.05, align='edge', edgecolor='black')

    # 设置 x 轴刻度以对应区间中点或边界（可选）
    plt.xticks(np.arange(0, 1.01, 0.1))  # 每 0.1 显示一个刻度
    plt.xlabel('Δ Attention')
    plt.ylabel('Frequency')
    plt.title('')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("./output_MIL/MixMHCpred_test/BeitAI-pHLA_result/attention_dist.png", dpi=500)

if __name__ == '__main__':
    # merge_data_MA()
    # merge_data_MixMHCpred()
    # merge_data_SA()
    # calc_MA_auc()
    # calc_MixMHCpred_auc()
    # calc_SA_auc()
    calc_BeitAI_attention_distribution()