import pandas as pd
import os
import matplotlib.pyplot as plt

# Follow https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments instructions and run FeatureExtraction
# Put this file in the same directory as the processed folder, together with result_x.csv

def main():
    result = []
    for i in range(0):

        path = "result_"+str(i)+".csv"

        df = pd.read_csv(path)
        info = df.iloc[: , :5]
        eye_gaze = df.iloc[: , 5:13]
        eye_2d = df.iloc[: , 13:125]
        eye_3d = df.iloc[: , 125:293]
        pose = df.iloc[: , 293:299]
        location_2d = df.iloc[: , 299:435]
        location_3d = df.iloc[: , 435:639]
        rigid = df.iloc[: , 639:645]
        non_rigid = df.iloc[: , 645:679]
        AU_r = df.iloc[: , 679:696]
        AU_c = df.iloc[: , 696:714]
        figure = True
        if not os.path.exists('graph'):
            os.mkdir('graph')
        if figure:
            fig, axs = plt.subplots(4, 4,figsize=(15, 15))
            l = ['Inner Brow Raiser','Outer Brow Raiser','Brow Lowerer','Upper Lid Raiser','Cheek Raiser',
                'Lid Tightener','Nose Wrinkler','Upper Lip Raiser','Lip Corner Puller',
                'Dimpler','Lip Corner Depressor','Chin Raiser',
                'Lip stretcher','Lip Tightener','Lips part',
                'Jaw Drop','Blink']
            for m in range(4):
                for j in range(4):
                    axs[m, j].plot(AU_r.iloc[: , m*4+j])
                    axs[m, j].set_title(l[m*4+j])
            f_path = os.path.join("graph","result_"+str(i)+".png")
            fig.savefig(f_path)
        
        motion_measure = sum([sum(AU_r[k]) for k in AU_r])
        result.append(motion_measure)

    print(result)

if __name__ == '__main__':
    main()
    
