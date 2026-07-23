
import os
import mhcflurry
import warnings

import numpy as np

# from MA_Mixpred_calculate import calculate_metrics
warnings.filterwarnings('ignore')

def SA_predict(data_in, data_out):
    filenames = os.listdir(data_in)
    for file in filenames:
        peptides = []
        label = []

        allele = file.replace(".txt", "")
        with open(data_in + file, "r") as f_in:
            for line in f_in.readlines():
                linelist = line.strip().split('\t')
                peptides.append(linelist[0])
                label.append(int(linelist[1]))

        results = predictor.predict(peptides=peptides, alleles=[allele], verbose=0)
        results['pred_binary'] = results['presentation_percentile'].apply(lambda x: 1 if x < 2 else 0)

        df_selected = results[['peptide', 'presentation_percentile','pred_binary','presentation_score']]
        df_selected.to_csv(data_out + file, sep='\t', index=False, header=False)

def MA_predict():
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
    
    data_in = "./data_random/MA_test/original/"
    data_out = "./output_MIL/MA_test/MHCflurry_result/"
    os.makedirs(data_out, exist_ok=True)
    filenames = os.listdir(data_in)

    for file in filenames:
        peptides = []
        label = []
    
        alleles = file.replace(".txt", "")
        alleles_list = allele_mhc_dict[alleles].strip().split(',')
        with open(data_in + file, "r") as f_in:
            for line in f_in.readlines():
                linelist = line.strip().split('\t')
                peptides.append(linelist[0])
                label.append(int(linelist[1]))
    
        results = predictor.predict(peptides=peptides, alleles=alleles_list, verbose=0)
        results['pred_binary'] = results['presentation_percentile'].apply(lambda x: 1 if x < 2 else 0)
        pred = results['pred_binary'].tolist()
        prob = results['presentation_score'].tolist()

        df_selected = results[['peptide', 'presentation_percentile','pred_binary','presentation_score']]

        df_selected.to_csv(data_out + file, sep='\t', index=False, header=False)



if __name__ == '__main__':
    predictor = mhcflurry.Class1PresentationPredictor.load()

    file_in1 = "./data_random/SA_test/HLA_seen/"
    file_in2 = "./data_random/SA_test/HLA_unseen/"
    file_out1 = "./output_MIL/SA_test_seen/MHCflurry_result/"
    file_out2 = "./output_MIL/SA_test_unseen/MHCflurry_result/"
    # MA_predict()
    SA_predict(file_in1, file_out1)
    SA_predict(file_in2, file_out2)