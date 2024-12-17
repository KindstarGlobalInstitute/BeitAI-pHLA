# BeitAI-pHLA

A peptide-HLA Binding Estimation using Immune Technology of Artificial Intelligence.

## Requirements

* numpy==1.26.4
* pandas==2.2.1
* torch==2.2.1
* pytorch-cuda==12.1
* fair-esm==2.0.0

## Installation

1. Download the BeitAI-pHLA repository.
2. Download the model weight file from https://drive.google.com/drive/folders/1TZ5crvmMiKCRJLJsdYxzrU89jTVMh7ERand and put it in `model_file/`.
3. Ensure that Python 3.x is installed.
4. Installation dependencies:
    ```
    pip install -r requirements.txt
    ```
5. Run model inference script:
    ```
    python predict.py -i test_data/input_data.txt 
    python predict.py -i test_data/input_data.txt -o output/output.txt
    python predict.py -i test_data/input_data.txt -o output/output.txt -d cuda:0
    python predict.py -i test_data/input_data.txt -o output/output.txt -d cuda:0 -b 128
    ```

## Options

- `-i, --input`: Path to the input data file..
- `-o, --output`: Path to save the output results..
- `-d, --device`: Whether to use GPU for inference(e.g., `cuda:0`).
- `-b, --bacth_size`: Batch size of input data.

## Support

If you have any questions or need further assistance, please contact [qsg6038@163.com] or visit the [GitHub Issues] page.
