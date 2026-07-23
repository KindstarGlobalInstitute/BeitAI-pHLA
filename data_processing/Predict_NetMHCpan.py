import subprocess
import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

allele_mhc_dict = {}
allelelist_path = './data_random/allelelist'
with open(allelelist_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        try:
            allele_name, mhc_name = line.split("\t")
        except:
            print(line)
        allele_mhc_dict[allele_name.strip()] = mhc_name.strip()

with open('./data_random/HLA_test.txt', "r") as f:
    for line in f.readlines():
        hla_name, hla_set = line.strip().split('\t', 1)
        hla_set_list = hla_set.strip().split('\t')
        allele_mhc_dict[hla_name] = ','.join(hla_set_list)


def run_command(cmd, index):
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return index, True, result.stdout, None
    except subprocess.CalledProcessError as e:
        return index, False, None, e.stderr

def run_NetMHCpan():
    commands = []
    # NOTE: Update this path to your local netMHCpan installation
    root = "~/netMHCpan/netMHCpan-4.1"
    os.chdir(root)

    file_in1 = "./data_random/SA_test/HLA_seen/"
    file_in2 = "./data_random/SA_test/HLA_unseen/"
    file_out1 = "./output_MIL/SA_test_seen/NetMHCpan_result/"
    file_out2 = "./output_MIL/SA_test_unseen/NetMHCpan_result/"
    
    os.makedirs(file_out1, exist_ok=True)
    os.makedirs(file_out2, exist_ok=True)

    command = "./netMHCpan -p ##input## -BA -xls -a ##allele## -xlsfile ##out_name##"
    start_time = time.time()

    for file in os.listdir(file_in1):
        HLA_name = file.replace('.txt', '')
        HLA_out_name = HLA_name.replace(':', '')
        # mhc_str = allele_mhc_dict[HLA_name]
        if not os.path.exists(file_out1 + HLA_out_name + "_out.txt"):
            command_temp = command.replace("##input##", file_in1 + file).replace("##allele##", HLA_name).replace("##out_name##", file_out1 + HLA_out_name + "_out.txt")
            commands.append(command_temp)
    
    for file in os.listdir(file_in2):
        HLA_name = file.replace('.txt', '')
        HLA_out_name = HLA_name.replace(':', '')
        # mhc_str = allele_mhc_dict[HLA_name]
        if not os.path.exists(file_out2 + HLA_out_name + "_out.txt"):
            command_temp = command.replace("##input##", file_in2 + file).replace("##allele##", HLA_name).replace("##out_name##", file_out2 + HLA_out_name + "_out.txt")
            commands.append(command_temp)

    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_index = {executor.submit(run_command, cmd, i): i for i, cmd in enumerate(commands)}
        
        for future in as_completed(future_to_index):
            index, success, stdout, stderr = future.result()
            results.append((index, success, stdout, stderr))
            print(f"已完成: {len(results)} / {len(commands)}")

    success_count = sum(1 for r in results if r[1])
    fail_count = len(commands) - success_count
    print(f"成功: {success_count}, 失败: {fail_count}")
    end_time = time.time()
    print("总运行时间:", end_time - start_time, "秒")

def netmhc_data_input(out_path):
    data = pd.read_csv("./output_MIL/MixMHCpred_test/MixMHCpred_data_result.csv", header=0)
    os.makedirs(out_path, exist_ok=True)
    data['NetMHCpan_pred'] = (data['NetMHCpan4.1'] < 2).astype(int)
    data_pos = data[data['NetMHCpan_pred'] == 1]

    # 按sampleName分组保存
    sample_list = list(data_pos['sampleName'].unique())
    for sample in sample_list:
        sample_data = data_pos[data_pos['sampleName'] == sample]
        sample_data = sample_data[['epitope', 'NetMHCpan_pred']]
        sample_data.to_csv(out_path + sample + ".txt", index=False, header=None, sep=" ")


def run_NetMHCpan_MixMHCpred():
    pep_path = "./data_random/MixMHCpred_test/original/"
    netmhc_data_input(pep_path)

    commands = []
    # NOTE: Update this path to your local netMHCpan installation
    root = "~/netMHCpan/netMHCpan-4.1"
    os.chdir(root)
    file_out = "./output_MIL/MixMHCpred_test/NetMHCpan_result_pos/"
    os.makedirs(file_out, exist_ok=True)

    command = "./netMHCpan -p ##input## -BA -xls -a ##allele## -xlsfile ##out_name##"
    start_time = time.time()

    for file in os.listdir(pep_path):
        allele_name = file.replace('.txt', '')
        hla_str = allele_mhc_dict[allele_name]
        if not os.path.exists(file_out + allele_name + ".out"):
            command_temp = command.replace("##input##", pep_path + file).replace("##allele##", hla_str).replace("##out_name##", file_out + allele_name + ".out")
            commands.append(command_temp)

    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_index = {executor.submit(run_command, cmd, i): i for i, cmd in enumerate(commands)}
        
        for future in as_completed(future_to_index):
            index, success, stdout, stderr = future.result()
            results.append((index, success, stdout, stderr))
            
            print(f"已完成: {len(results)} / {len(commands)}")

    success_count = sum(1 for r in results if r[1])
    fail_count = len(commands) - success_count
    print(f"成功: {success_count}, 失败: {fail_count}")
    end_time = time.time()
    print("总运行时间:", end_time - start_time, "秒")

def NetMHCpan_MA_static(file_in, file_out):
    '''
    @Desc   : 统计NetMHCpan计算结果(MA) 
    '''
    os.makedirs(file_out, exist_ok=True)

    filenames = os.listdir(file_in)
    for filename in filenames:
        # mhc_name = filename.replace(".out", "")
        mhc_name = filename.replace("_out.txt", "")
        mhc_name = mhc_name[:-2] + ':' + mhc_name[-2:]

        df = pd.read_csv(file_in + filename, sep='\t', skiprows=1, header=0)

        col_name = df.columns.tolist()
        EL_rank_col = []
        for i in col_name:
            if i.startswith('EL_Rank'):
                EL_rank_col.append(i)
        df['NB'] = df['NB'].apply(lambda x: 0 if x == 0 else 1)
        df['EL_Rank_all'] = df[EL_rank_col].min(axis=1)
        df_selected = df[['Peptide', 'EL_Rank_all', 'NB']]
        df_selected.to_csv(file_out + mhc_name + '.txt', sep='\t', index=False, header=False)

if __name__ == '__main__':
    # run_NetMHCpan()
    run_NetMHCpan_MixMHCpred()

    # file_in1 = "./output_MIL/SA_test_seen/NetMHCpan_result/"
    # file_in2 = "./output_MIL/SA_test_unseen/NetMHCpan_result/"
    # file_out1 = "./output_MIL/SA_test_seen/NetMHCpan_result_static/"
    # file_out2 = "./output_MIL/SA_test_unseen/NetMHCpan_result_static/"
  
    # NetMHCpan_MA_static(file_in1, file_out1)
    # NetMHCpan_MA_static(file_in2, file_out2)
