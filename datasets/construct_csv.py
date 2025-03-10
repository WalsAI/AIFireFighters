import pandas as pd
import os

class ConstructCSV:
    def __init__(self, path, csv_name):
        self.path = path
        self.__dict = {
            "image": [],
            "label": [],
            "step": [],
        }
        self.csv_name = csv_name

    def __construct_csv(self):
        split = os.listdir(self.path)
        for i in split:
            for j in os.listdir(os.path.join(self.path, i)):
                for k in os.listdir(os.path.join(self.path, i, j)):
                    self.__dict["image"].append(os.path.join(i, j, k))
                    if 'no' in j.lower():
                        self.__dict["label"].append(0)
                    else:
                        self.__dict["label"].append(1)
                    if 'train' in i.lower():
                        self.__dict["step"].append("train")
                    elif 'val' in i.lower() or 'test' in i.lower():
                        self.__dict["step"].append("val")
        pd.DataFrame(self.__dict).to_csv(self.csv_name, index=False)
    
    def __call__(self):
        self.__construct_csv()

if __name__ == "__main__":
    ConstructCSV("/raid/bigdata/userhome/ionut.serban/sharedData/controlnet_mirpr/AIFireFighters/datasets/wildfire_dataset", "wildfire_dataset.csv")()
    ConstructCSV("/raid/bigdata/userhome/ionut.serban/sharedData/controlnet_mirpr/AIFireFighters/datasets/forest_dataset", "forest_dataset.csv")()