import pandas as pd
import numpy as np
import sys


def main(symps):
    dis_prob = pd.read_csv('C:\Development\__SCHOOL__\ml_homeworks\\baies\disease.csv', sep=';')
    dis_prob['probability'] = dis_prob['количество пациентов'] / dis_prob['количество пациентов'].iloc[-1]
    dis_prob.drop(dis_prob.tail(1).index, inplace=True)
    #print(dis_prob)

    symp_prob = pd.read_csv('C:\Development\__SCHOOL__\ml_homeworks\\baies\symptom.csv', sep=';')
    #print(symp_prob)

    #symps = np.random.choice(a=[False, True], size=len(symp_prob))
    #print(symps)

    #symps = [True, False, True, True, False, True, True, False, False, True, False, True, False, False, True, True,
    #         True, True, False, False, False, False, False]

    dis_with_symps_prob = np.ones(len(dis_prob))
    for i in range(len(dis_prob)):
        for j, symp in enumerate(symp_prob):
            if symps[j]:
                dis_with_symps_prob[i] *= symp_prob.iloc[j][i + 1]
                #print(symp_prob.iloc[j][i + 1])
        dis_with_symps_prob[i] *= dis_prob['probability'][i]
        #print(dis_prob['probability'][i])
    #print(dis_with_symps_prob)

    index = -1
    max = -1
    for i in range(len(dis_with_symps_prob)):
        if dis_with_symps_prob[i] > max:
            max = dis_with_symps_prob[i]
            index = i
    #print("\nСимптомы:")
    #for i in range(len(symps)):
        #if symps[i]:
            #print(symp_prob.iloc[i][0])
    #print("\nБолезнь:")
    print(dis_prob.iloc[index][0])


if __name__ == '__main__':
    vals = []
    for i in range(len(sys.argv)):
        val = sys.argv[i] == 'true' if True else False
        vals.append(val)
    main(vals)
