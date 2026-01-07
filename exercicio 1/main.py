import pandas as pd
from data_analisis import data_analisis
from pre_processing import pre_processing

def main():
    datafile = pd.read_csv('../SMSSpamCollection', sep = '\t', names=['label', 'text'])
    
    data_analisis(datafile)

    df_limpo = pre_processing(datafile)

if __name__ == "__main__":
    main()