import ast
import os
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from statannotations.Annotator import Annotator

# NOTE: Update these paths to match your local environment
python_path = '~/anaconda3/envs/python2.7/bin/python'
seq2logo_path = '~/seq2logo-2.0/Seq2Logo.py'
rootpath = "./output_MIL/MixMHCpred_test/motif_analyze/"

def get_HLA_dict():
    # 加载HLA类型
    allele_HLA_dict = {}
    with open("./data_random/HLA_test.txt", 'r') as f:
        for line in f:
            line = line.strip()
            name, allele = line.split('\t', maxsplit=1)
            allele_list = allele.split("\t")
            allele_HLA_dict[name] = allele_list
    return allele_HLA_dict

def exec_seq2logo_single(pep_path, out_path, python_path, seq2logo_path):
    out_name = pep_path.split('/')[-1].replace('.txt', '')
    # out_name = allele + '-' + name
    # command = [python_path, seq2logo_path, '-f', pep_path, '-o', out_path + out_name, '--colors', 'C7522A:DE,E5C185:QSTYNG,008585:HKR', '-b', '0', '-C', '1','--format', 'PDF']
    command = [python_path, seq2logo_path, '-f', pep_path, '-o', out_path + out_name, '--colors', 'C7522A:DE,E5C185:QSTYNG,008585:HKR', '-b', '0', '-C', '1','--format', 'PDF']
    # command = [python_path, seq2logo_path, '-f', pep_path + file, '-o', out_path + HLA_name + '_0.5', '-b','0', '-C', '1','--format', 'PDF']

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print("Script executed successfully.")
    else:
        print(f"Script failed with return code {result.returncode}.")


def read_logo_file(file):
    '''
    @Desc   : 读取Seq2logo计算结果
    '''
    # 从第四行读取
    df = pd.read_csv(file, sep=' ', skiprows=3, header=None)
    res_list = []
    for raw in df.iterrows():
        raw_list = raw[1].tolist()
        res_list += raw_list[2:]
    return res_list

def _best_hla(row, cols, hla_list):
        lists = [list(map(float, row[col].split(','))) for col in cols]
        avg_list = np.mean(lists, axis=0)
        max_idx = np.argmax(avg_list)
        return hla_list[max_idx]


def get_NetMHCpan_pep():
    file_in = "./output_MIL/MixMHCpred_test/NetMHCpan_result_pos/" 
    for pep_len in [8, 9, 10, 11]:
        file_out = f"{rootpath}NetMHCpan_pep/len_" + str(pep_len) + "/"
        os.makedirs(file_out, exist_ok=True)
        # pep_res_dict = {}
        filenames = os.listdir(file_in)
        for filename in filenames:
            pep_res_dict = {}
            allele_name = filename.replace(".out", "")
            # 读取filename第一行
            with open(file_in + filename, "r") as f:
                first_line = f.readline()
            first_line = first_line.strip().split("\t\t\t\t\t\t")
            df = pd.read_csv(file_in + filename, sep='\t', skiprows=1, header=0)
            df["len"] = df['Peptide'].str.len()
            df = df[df['len'] == pep_len]
            col_name = df.columns.tolist()
            EL_rank_col = []
            for i in col_name:
                if i.startswith('EL_Rank'):
                    EL_rank_col.append(i)
            EL_score = df[['Peptide'] + EL_rank_col]
            # 更改EL_score列名
            EL_score.columns = ['Peptide'] + first_line
            EL_score["best_index"] = EL_score[first_line].idxmin(axis=1)

            # 按best_index分组
            for name, group in EL_score.groupby(["best_index"]):
                hla_name = name[0]
                pep_list = group['Peptide'].tolist()
                pep_res_dict[hla_name] = pep_list

            for hla_name, pep_list in pep_res_dict.items():
                hla_outname = hla_name.replace("HLA-", "").replace(":", "")
                with open(file_out + allele_name + "_"+ hla_outname + ".txt", "w") as f:
                    for pep in pep_list:
                        f.write(pep + "\n")


def get_motif_atlas_pep():
    '''
    @Desc   : 从MotifAtlas数据库中获取 motif 信息
    '''
    all_HLA_alias = []
    for HLA in all_HLA:
        hla = HLA.replace("HLA-", "").replace(":", "")
        all_HLA_alias.append(hla)


    file_in = "./data_random/MotifAtlas_all_peptides.txt" 
    atlas_data = pd.read_csv(file_in, sep='\t')
    atlas_data = atlas_data[atlas_data['Allele'].isin(all_HLA_alias)]

    for pep_len in [8, 9, 10, 11]:
        file_out = f"{rootpath}MotifAtlas_pep/len_" + str(pep_len) + "/"
        os.makedirs(file_out, exist_ok=True)
        atlas_data_tmp = atlas_data[atlas_data['Peptide'].str.len() == pep_len]

        # 按best_index分组
        for name, group in atlas_data_tmp.groupby(["Allele"]):
            group_filtered = group[['Peptide']]
            group_filtered.to_csv(f"{file_out}{name[0]}.txt", sep="\t", index=False, header=False)

def get_BeitAI_pep():

    pep_path = f"{rootpath}BeitAI_pep/"
    os.makedirs(pep_path, exist_ok=True)
    # pep_res_dict = {}
    for pep_len in [8, 9, 10, 11]:
        file_out = pep_path + "len_" + str(pep_len) + "/"
        os.makedirs(file_out, exist_ok=True)
        for filename in allele_HLA_dict.keys():
            pep_res_dict = {}
            HLA_list = allele_HLA_dict[filename]
            f_df = pd.read_csv(f"./output_MIL/MixMHCpred_test/BeitAI-pHLA_result/attention_result/{filename}.txt", sep="\t", index_col=False, header=None, names=["pep", "label", "prob_0", "att_list_0", "prob_1", "att_list_1", "prob_2", "att_list_2"])
            f_df['len'] = f_df['pep'].str.len()

            f_df['prob'] = f_df[['prob_0', 'prob_1', 'prob_2']].mean(axis=1)
            f_df['pred'] = f_df.apply(lambda row: 1 if row['prob'] >= 0.5 else 0, axis=1)
            f_df_filter = f_df[(f_df['len'] == pep_len) & (f_df["pred"] == 1)]
            # f_df_filter['best_hla'] = f_df_filter['att_list'].apply(lambda s: HLA_list[np.argmax(list(map(float, s.split(','))))])
            f_df_filter['best_hla'] = f_df_filter.apply(lambda row: _best_hla(row, ["att_list_0", "att_list_1", "att_list_2"], HLA_list),axis=1)

            # 按best_index分组
            for name, group in f_df_filter.groupby(["best_hla"]):
                hla_name = name[0]
                pep_list = group['pep'].tolist()
                pep_res_dict[hla_name] = pep_list

            for hla_name, pep_list in pep_res_dict.items():
                hla_outname = hla_name.replace("HLA-", "").replace(":", "")
                if len(pep_list) >= 100:
                    with open(file_out + filename + "_"+ hla_outname + ".txt", "w") as f:
                        for pep in pep_list:
                            f.write(pep + "\n")

def exec_seq2logo(pep_path, out_path):
    allele_list = os.listdir(pep_path)
    for file in allele_list:
        out_name = file.replace('.txt', '')
        command = [python_path, seq2logo_path, '-f', pep_path + file, '-o', out_path + out_name, '--colors', 'C7522A:DE,E5C185:QSTYNG,008585:HKR', '-b', '0', '-C', '1','--format', 'PDF']
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print("Script executed successfully.")
        else:
            print(f"Script failed with return code {result.returncode}.")

def logo_plot():
    for pep_len in [8, 9, 10, 11]:
        # BeitAI可视化结果
        fout_pep = f"{rootpath}BeitAI_pep/len_" + str(pep_len) + "/"
        fout_res = f"{rootpath}BeitAI_logo/len_" + str(pep_len) + "/"
        os.makedirs(fout_res, exist_ok=True)
        exec_seq2logo(fout_pep, fout_res)

        # # NetMHCpan可视化
        # fout_pep = f"{rootpath}NetMHCpan_pep/len_" + str(pep_len) + "/"
        # fout_res = f"{rootpath}NetMHCpan_logo/len_" + str(pep_len) + "/"
        # os.makedirs(fout_res, exist_ok=True)
        # exec_seq2logo(fout_pep, fout_res)
    
        # # MotifAtlas可视化结果
        # fout_pep = f"{rootpath}MotifAtlas_pep/len_" + str(pep_len) + "/"
        # fout_res = f"{rootpath}MotifAtlas_logo/len_" + str(pep_len) + "/"
        # os.makedirs(fout_res, exist_ok=True)
        # exec_seq2logo(fout_pep, fout_res)
    
def correlation_motif(res_type="pssm"):
    '''
    @Desc   : 计算MotifAtlas和NetMHCpan、BeitAI的相关性
    '''
    corr_BeitAI = []
    corr_NetMHCpan = []

    if res_type=="pssm":
        suffix = ".txt"
    elif res_type=="pfm":
        suffix = "_freq.mat"

    for pep_len in [8, 9, 10, 11]:
        # 读取NetMHCpan数据
        res_NetMHCpan = {}
        netmhc_path = f"{rootpath}NetMHCpan_logo/len_" + str(pep_len) + "/"
        file_list = os.listdir(netmhc_path)
        for file in file_list:
            if file.endswith(suffix):
                file_name = file.replace(suffix, "")
                name_list = file_name.rsplit('_', 1)
                allele_name = name_list[0]
                HLA_name = name_list[1]
                netmhc_data = read_logo_file(f'{netmhc_path}{file}')
                if allele_name not in res_NetMHCpan.keys():
                    res_NetMHCpan[allele_name] = {}
                res_NetMHCpan[allele_name][HLA_name] = netmhc_data

        # 读取BeitAI数据
        res_BeitAI = {}
        beitai_path = f"{rootpath}/BeitAI_logo/len_" + str(pep_len) + "/"
        file_list = os.listdir(beitai_path)
        for file in file_list:
            if file.endswith(suffix):
                file_name = file.replace(suffix, "")
                name_list = file_name.rsplit('_', 1)
                allele_name = name_list[0]
                HLA_name = name_list[1]
                beitai_data = read_logo_file(f'{beitai_path}{file}')
                if allele_name not in res_BeitAI.keys():
                    res_BeitAI[allele_name] = {}
                res_BeitAI[allele_name][HLA_name] = beitai_data

        # 读取MotifAtlas数据
        res_MotifAtlas = {}
        motif_path = f"{rootpath}/MotifAtlas_logo/len_" + str(pep_len) + "/"
        file_list = os.listdir(motif_path)
        for file in file_list:
            if file.endswith(suffix):
                HLA_name = file.replace(suffix, "")
                motif_data = read_logo_file(f'{motif_path}{file}')
                res_MotifAtlas[HLA_name] = motif_data


        # 计算相关性

        for allele_tmp in res_BeitAI.keys():
            for HLA_tmp in res_BeitAI[allele_tmp].keys():
                if allele_tmp in res_NetMHCpan.keys() and HLA_tmp in res_NetMHCpan[allele_tmp].keys() and HLA_tmp in res_MotifAtlas.keys():
                    netmhc_data = res_NetMHCpan[allele_tmp][HLA_tmp]
                    beitai_data = res_BeitAI[allele_tmp][HLA_tmp]
                    motif_data = res_MotifAtlas[HLA_tmp]

                    netmhc_temp = np.array(netmhc_data)
                    beitai_temp = np.array(beitai_data)
                    motif_temp = np.array(motif_data)

                    # # 创建有效索引：netmhc 和 beitai 都不等于 -99.999
                    # valid_netmhc_mask = (netmhc_temp != -99.999)
                    # valid_beitai_mask = (beitai_temp != -99.999)

                    # # 提取有效数据
                    # netmhc_valid = netmhc_temp[valid_netmhc_mask]
                    # beitai_valid = beitai_temp[valid_beitai_mask]
                    # motif_netmhc_valid = motif_temp[valid_netmhc_mask]
                    # motif_beitai_valid = motif_temp[valid_beitai_mask]

                    # # 计算相关性
                    netmhc_corr, p_netmhc = pearsonr(netmhc_temp, motif_temp)
                    beitai_corr, p_beitai = pearsonr(beitai_temp, motif_temp)
                    corr_BeitAI.append(beitai_corr)
                    corr_NetMHCpan.append(netmhc_corr)

    corr_file_pssm = f"{rootpath}corr_{res_type}.txt"
    with open(corr_file_pssm, 'w') as f:
        for i in range(len(corr_BeitAI)):
            f.write(f"{corr_BeitAI[i]}\t{corr_NetMHCpan[i]}\n")

def get_high_low_att_pep(HLA_name, outpath):
    os.makedirs(outpath, exist_ok=True)
    pep_list = []
    for filename in allele_HLA_dict.keys():
        HLA_list = allele_HLA_dict[filename]
        if HLA_name in HLA_list:
            index = HLA_list.index(HLA_name)
            f_df = pd.read_csv(f"./output_MIL/MixMHCpred_test/BeitAI-pHLA_result/attention_result/{filename}.txt", sep="\t", index_col=False, header=None, names=["pep", "label", "prob_0", "att_list_0", "prob_1", "att_list_1", "prob_2", "att_list_2"])
            f_df['len'] = f_df['pep'].str.len()
            f_df['prob'] = f_df[['prob_0', 'prob_1', 'prob_2']].mean(axis=1)
            f_df['pred'] = f_df.apply(lambda row: 1 if row['prob'] >= 0.5 else 0, axis=1)
            f_df_filter = f_df[(f_df['len'] == 9) & (f_df["pred"] == 1)]
            cols = ["att_list_0", "att_list_1", "att_list_2"]
            f_df_filter[HLA_name] = f_df_filter.apply(
                lambda row: np.mean([list(map(float, row[col].split(','))) for col in cols], axis=0)[index],
                axis=1
            )
            pep_list.append(f_df_filter[['pep', HLA_name]])
    if len(pep_list) == 0:
        print(f"{HLA_name} no pep!")
        return

    pep_df = pd.concat(pep_list, axis=0)

    df_sorted = pep_df.sort_values(by=HLA_name, ascending=False).reset_index(drop=True)
    # 2. 计算前10%和后10%的行数（向上取整避免0行）
    n = len(df_sorted)
    top10_pct = int(np.ceil(n * 0.1))  # 前10%的行数
    bottom10_pct = int(np.ceil(n * 0.1))  # 后10%的行数

    # 3. 提取前10%和后10%的数据
    top10_df = df_sorted.head(top10_pct)  # 升序排序后，前10%是最小值的部分
    bottom10_df = df_sorted.tail(bottom10_pct)  # 升序排序后，后10%是最大值的部分
    # 保存top10_df的pep列到txt文件
    top10_df['pep'].to_csv(f"{outpath}top10_{HLA_name}.txt", index=False, header=False)
    # 保存bottom10_df的pep列到txt文件
    bottom10_df['pep'].to_csv(f"{outpath}bottom10_{HLA_name}.txt", index=False, header=False)
    print("done!")

def plot_attention_motif(HLA_name, filepath, outpath):
    os.makedirs(outpath, exist_ok=True)
    # 调用Seq2Logo.py脚本绘制Logo图
    fout_pep_top10 = f"{filepath}top10_{HLA_name}.txt"
    fout_pep_bottom10 = f"{filepath}bottom10_{HLA_name}.txt"
    # fout_res = "~/MHCpred_MIL/data/MixMHCpred_test/motif_analyze/BeitAI_logo/"
    exec_seq2logo_single(fout_pep_top10, outpath, python_path, seq2logo_path)
    exec_seq2logo_single(fout_pep_bottom10, outpath, python_path, seq2logo_path)

def attention_correlation_motif():
    '''
    @Desc   : 计算MotifAtlas和BeitAI高注意力和低注意力的Logo图的相关性
    '''
    result_path = f"{rootpath}BeitAI_logo/top_bottom/"
    res_BeitAI_high = {}
    res_BeitAI_low = {}
    for file in os.listdir(result_path):
        # BeitAI_high
        if file.startswith("top10") and file.endswith("freq.mat"):
        # if file.startswith("top10") and file.endswith(".txt"):
            hla_name = file.split("_")[1].replace(".txt", "")
            hla_data = read_logo_file(f'{result_path}{file}')
            res_BeitAI_high[hla_name] = hla_data
        # BeitAI_low
        elif file.startswith("bottom10") and file.endswith("freq.mat"):
        # elif file.startswith("bottom10") and file.endswith(".txt"):
            hla_name = file.split("_")[1].replace(".txt", "")
            hla_data = read_logo_file(f'{result_path}{file}')
            res_BeitAI_low[hla_name] = hla_data

    # 读取MotifAtlas数据
    res_MotifAtlas = {}
    motif_path = f"{rootpath}MotifAtlas_logo/len_9/"
    file_list = os.listdir(motif_path)
    for file in file_list:
        if file.endswith("freq.mat"):
        # if file.endswith(".txt"):
        #     HLA_name = file.replace(".txt", "")
            HLA_name = file.replace("_freq.mat", "")
            # HLA_name第四个位置插入：_
            HLA_name = "HLA-" + HLA_name[:3] + ":" + HLA_name[3:]
            motif_data = read_logo_file(f'{motif_path}{file}')
            res_MotifAtlas[HLA_name] = motif_data

    # 计算相关性
    correlation_res = {}
    for HLA_name in all_HLA:
        BeitAI_high_data = res_BeitAI_high[HLA_name]
        BeitAI_low_data = res_BeitAI_low[HLA_name]
        motif_data = res_MotifAtlas[HLA_name]

        BeitAI_high_temp = np.array(BeitAI_high_data)
        BeitAI_low_temp = np.array(BeitAI_low_data)
        motif_temp = np.array(motif_data)

        # # 创建有效索引：netmhc 和 beitai 都不等于 -99.999
        # valid_BeitAI_high_mask = (BeitAI_high_temp != -99.999)
        # valid_BeitAI_low_mask = (BeitAI_low_temp != -99.999)

        # # 提取有效数据
        # BeitAI_high_valid = BeitAI_high_temp[valid_BeitAI_high_mask]
        # BeitAI_low_valid = BeitAI_low_temp[valid_BeitAI_low_mask]
        # motif_BeitAI_high_valid = motif_temp[valid_BeitAI_high_mask]
        # motif_BeitAI_low_valid = motif_temp[valid_BeitAI_low_mask]
        # 计算相关性
        # BeitAI_high_corr, p_BeitAI_high = pearsonr(BeitAI_high_valid, motif_BeitAI_high_valid)
        # BeitAI_low_corr, p_BeitAI_low = pearsonr(BeitAI_low_valid, motif_BeitAI_low_valid)
        BeitAI_high_corr, p_BeitAI_high = pearsonr(BeitAI_high_temp, motif_temp)
        BeitAI_low_corr, p_BeitAI_low = pearsonr(BeitAI_low_temp, motif_temp)
        correlation_res[HLA_name] = [BeitAI_high_corr, BeitAI_low_corr]

    high_corr_all = [correlation_res[HLA_name][0] for HLA_name in all_HLA]
    low_corr_all = [correlation_res[HLA_name][1] for HLA_name in all_HLA]

    return high_corr_all, low_corr_all

def plot_attention_correlation():
    colors = ['#963e20', '#006464']
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=300)
    fig, axs = plt.subplots(figsize=(5, 6), dpi=300)

    high_corr, low_corr = attention_correlation_motif()

    # 子图位置和数据框的映射
    pairs = [("High_attention_score", "Low_attention_score")]
    data = pd.DataFrame({
        pairs[0][0]: high_corr,
        pairs[0][1]: low_corr
    })
    sns.boxplot(data=data, color="white", width=0.4, linewidth=1.2, showfliers=False, ax=axs, zorder=1)
    # 设置颜色
    for j, box in enumerate(axs.patches):
        if j < len(colors):
            box.set_edgecolor(colors[j])
            box.set_linewidth(1.2)
    # 设置线条颜色
    lines = axs.get_lines()
    for k in range(len(axs.patches)):
        color_idx = k % len(colors)
        current_color = colors[color_idx]
        line_start_idx = k * 5 
        for line_offset in range(5): 
            line_idx = line_start_idx + line_offset
            if line_idx < len(lines):
                lines[line_idx].set_color(current_color)
                lines[line_idx].set_linewidth(1.2)

    sns.stripplot(data=data, palette=colors, size=6, alpha=0.8, ax=axs, zorder=10)

    annot = Annotator(axs, pairs, data=data)
    annot.configure(test='Mann-Whitney', text_format='full', loc='inside', verbose=1, line_height=0.01,
                    line_width=0.7, show_test_name=False, pvalue_format_string="{:.2e}")
    annot.apply_test()

    _, test_results = annot.annotate()

    # 显示图形
    # axs.margins(x=0.12)
    axs.set_ylabel('Correlation')
    axs.set_xlabel('')

    for spine in ['top', 'bottom', 'left', 'right']:
        axs.spines[spine].set_linewidth(1)
    plt.subplots_adjust(left=0.15, right=0.92, bottom=0.2, top=0.9)
    # plt.show()
    plt.savefig(f"{rootpath}BeitAI_Attention_PFM.eps", format='eps')

if __name__ == '__main__':
    
    allele_HLA_dict = get_HLA_dict()
    all_HLA = []
    for HLA_list in allele_HLA_dict.values():
        all_HLA += HLA_list
    all_HLA = list(set(all_HLA))
    # # 按HLA和肽段长度获取阳性肽段
    # get_NetMHCpan_pep()
    # get_BeitAI_pep()
    # get_motif_atlas_pep()

    # 输出seq2logo文件
    # logo_plot()

    # 计算相关性
    # correlation_motif("pssm")
    # correlation_motif("pfm")

    # # 计算高/低注意力PFM
    # pep_path = f"{rootpath}BeitAI_pep/"
    # logo_path = f"{rootpath}BeitAI_logo/"
    # for HLA_name in all_HLA:
    #     get_high_low_att_pep(HLA_name, pep_path + "top_bottom/")
    #     plot_attention_motif(HLA_name, pep_path + "top_bottom/", logo_path + "top_bottom/")

    plot_attention_correlation()