import numpy as np
import pandas as pd


def main():
    dis_prob = pd.read_csv("disease.csv", sep=";")
    dis_prob['probability'] = dis_prob['количество пациентов'] / dis_prob['количество пациентов'].iloc[-1]
    dis_prob.drop(dis_prob.tail(1).index, inplace=True)
    print(dis_prob)


if __name__ == '__main__':
    main()
