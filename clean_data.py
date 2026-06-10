import os
import glob
import shutil
from tqdm import tqdm

def data_process(path = os.getcwd(), remove = False, giga = False, check = True):
    if not os.path.exists(path):
        raise FileNotFoundError("Idiot there is no folder like you mention")
    os.makedirs(path+"/data", exist_ok = True)
    for i in glob.glob(path + "/**/*fr-en.*", recursive = True):
        if "annotation" not in i:
            os.rename(i, path+"/data/"+os.path.basename(i))
    if giga:
        os.rename("/teamspace/studios/this_studio/train/giga-fren.release2.fixed.en", "/teamspace/studios/this_studio/data/giga-fren.release2.fr-en.en")
        os.rename("/teamspace/studios/this_studio/train/giga-fren.release2.fixed.fr", "/teamspace/studios/this_studio/data/giga-fren.release2.fr-en.fr")
    if remove:
        shutil.rmtree(path+"/train/")
    if check:
        redundant = []
        for i in tqdm(glob.glob(path + "/data/*.en")):
            with open(i, 'r') as f1, open(i[:-2] + "fr") as f2:
                d1, d2 = sum(1 for _ in f1), sum(1 for _ in f2)
            if d1 != d2:
                redundant.append(i)
                redundant.append(i[:-2]+'fr')
        if len(redundant):
            print("---------------------redundant Files--------------------")
            print("\n".join([os.path.basename(i) for i in redundant]))
            print("_________________________________________________________")
            ui = input("Y for remove redunant files, any other keys to continue: ").lower()
            if ui == "y":
                for i in redundant:
                    os.remove(i)
            
if __name__ == "__main__":
    data_process()


    

