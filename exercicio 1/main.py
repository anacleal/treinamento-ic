import pandas as pd
from data_analisis import data_analisis
from modeling import modeling

def main():
    datafile = pd.read_csv('../SMSSpamCollection', sep = '\t', names=['label', 'text'])
    
    #datafile = data_analisis(datafile)

    datafile = modeling(datafile)

if __name__ == "__main__":
    main()