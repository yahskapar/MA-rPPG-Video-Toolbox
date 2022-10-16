import os
import mat73
import pandas as pd

def main():
    
    path_gr = "P000001.mat"
    mat = mat73.loadmat(path_gr)
    ground_truth_au2 = mat['au2']  # load raw frames
    ground_truth_au4 = mat['au4']
    ground_truth_au7 = mat['au7']
    ground_truth_au12 = mat['au12']
    ground_truth_au15 = mat['au15']
    ground_truth_au17 = mat['au17']
    ground_truth_au18 = mat['au18']
    ground_truth_au26 = mat['au26']
    ground_truth_au27 = mat['au27']
    ground_truth_au45 = mat['au45']
    
    path_o = "test_p1.csv"
    df = pd.read_csv(path_o)
    AU_r = df.iloc[: , 679:696]
    #AU_c = df.iloc[: , 696:714]
    #print(AU_r["AU02_r"])
    #[AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r] = AU_r
    size = len(ground_truth_au2)
    '''
    diff_au2  = sum([abs(ground_truth_au2[i]-AU02_r[i]/5) for i in range(size)])
    diff_au4  = sum([abs(ground_truth_au4[i]-AU04_r[i]/5) for i in range(size)])
    diff_au7  = sum([abs(ground_truth_au7[i]-AU07_r[i]/5) for i in range(size)])
    diff_au12 = sum([abs(ground_truth_au12[i]-AU12_r[i]/5) for i in range(size)])
    diff_au15 = sum([abs(ground_truth_au15[i]-AU15_r[i]/5) for i in range(size)])
    diff_au17 = sum([abs(ground_truth_au17[i]-AU17_r[i]/5) for i in range(size)])
    #diff_au18 = sum([abs(ground_truth_au18[i]-AU18_r[i]/5) for i in range(size)])
    diff_au26 = sum([abs(ground_truth_au26[i]-AU26_r[i]/5) for i in range(size)])
    #diff_au27 = sum([abs(ground_truth_au27[i]-AU27_r[i]/5) for i in range(size)])
    diff_au45 = sum([abs(ground_truth_au45[i]-AU45_r[i]/5) for i in range(size)])
    '''
    diffs = [
    sum([abs(ground_truth_au2[i]-AU_r["AU02_r"][i])/5 for i in range(size)]),
    sum([abs(ground_truth_au4[i]-AU_r["AU04_r"][i]/5) for i in range(size)]),
    sum([abs(ground_truth_au7[i]-AU_r["AU07_r"][i]/5) for i in range(size)]),
    sum([abs(ground_truth_au12[i]-AU_r["AU12_r"][i]/5) for i in range(size)]),
    sum([abs(ground_truth_au15[i]-AU_r["AU15_r"][i]/5) for i in range(size)]),
    sum([abs(ground_truth_au17[i]-AU_r["AU17_r"][i]/5) for i in range(size)]),
    sum([abs(ground_truth_au26[i]-AU_r["AU26_r"][i]/5) for i in range(size)]),
    sum([abs(ground_truth_au45[i]-AU_r["AU45_r"][i]/5) for i in range(size)]),
    ]

    for i in diffs:
        print(i)
    
    
    
    

if __name__ == '__main__':
    main()
    