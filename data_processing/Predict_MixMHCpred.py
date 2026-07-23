import os
import pandas as pd
import subprocess

ROOT_PATH = "./output_MIL/MA_test/"

MHC_pseudo_dict = {}
allele_mhc_dict = {}
pseudo_path = './data_random/MHC_pseudo.dat'
allelelist_path = './data_random/allelelist'
with open(pseudo_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        MHC_name, pseudo_str = line.split(" ")
        MHC_pseudo_dict[MHC_name.strip()] = pseudo_str.strip()

with open(allelelist_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        try:
            allele_name, mhc_name = line.split("\t")
        except:
            print(line)
        allele_mhc_dict[allele_name.strip()] = mhc_name.strip()


def MixMHCpred_sh():
    file_pep = "./data_random/MA_test/original_NoLabel/"
    outpath = "./output_MIL/MA_test/MixMHCpred_result_2.2/"
    os.makedirs(outpath, exist_ok=True)

    # NOTE: Update this path to your local MixMHCpred installation
    root_str = "~/MixMHCpred-2.2/MixMHCpred -i #input# -o #output# -a #allele#"
    filelist = os.listdir(file_pep)
    # filelist = ["Bcell.txt"]
    count = 0
    for filename in filelist:
        count += 1
        mhc_name = filename.replace(".txt", '')
        mhc_str = allele_mhc_dict[mhc_name]
        mhc_str = mhc_str.replace("HLA-", "").replace(":", "")

        input_str = os.path.join(file_pep, filename)
        output_str = os.path.join(outpath, filename)
        excut_str = root_str.replace("#input#", input_str).replace("#output#", output_str).replace("#allele#", mhc_str)
        print(excut_str)
        try:
            result = subprocess.run(excut_str, shell=True, check=True, capture_output=True, text=True)
            print(count)
        except subprocess.CalledProcessError as e:
            print(f"Command: {excut_str} failed with error:\n{e.stderr}\n")

def result_static():
    file_in = f"{ROOT_PATH}MixMHCpred_result_2.2/"
    file_out = f"{ROOT_PATH}MixMHCpred_result_2.2_static/"
    os.makedirs(file_out, exist_ok=True)
    filelist = os.listdir(file_in)
    for filename in filelist:
        print(f"Processing {filename}")
        df = pd.read_csv(file_in + filename, comment='#', sep='\t')
        df_selected = df[['Peptide', '%Rank_bestAllele']]
        df_selected.to_csv(file_out + filename, sep='\t', index=False, header=False)

if __name__ == '__main__':
    # MixMHCpred_sh()
    result_static()