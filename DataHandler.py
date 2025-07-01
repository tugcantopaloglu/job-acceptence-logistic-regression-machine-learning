import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
class DataHandler():
    #standart const
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        
    #Dosyadan okuma işlemleri    
    def DataScrapper(self):
        self.csv_data = pd.read_csv(self.data_file_path, header=None, names=["Exam1", "Exam2", "Admitted"])
        print("Data has been parsed succesfully...")
        
    #Veriyi eğitim için bölme (başka class'ta kullanırsak diye veriyi getter ile de alacağız)
    #tekrar tekrar bölmemek için dosyalara kaydedelim eğer dosyada yoksa bölsün
    def DataSplitter(self):
        if os.path.exists("./data/train_data.txt") and os.path.exists("./data/validate_data.txt") and os.path.exists("./data/test_data.txt"):
            self.train_data = pd.read_csv("./data/train_data.txt")
            self.validate_data = pd.read_csv("./data/validate_data.txt")
            self.test_data = pd.read_csv("./data/test_data.txt")
        else:
            self.train_data, self.temp_data = train_test_split(self.csv_data, test_size=0.4, random_state=42, stratify=self.csv_data["Admitted"])
            self.validate_data, self.test_data = train_test_split(self.temp_data, test_size=0.5, random_state=42, stratify=self.temp_data["Admitted"])
            self.train_data.to_csv("./data/train_data.txt", index=False)
            self.validate_data.to_csv("./data/validate_data.txt", index=False)
            self.test_data.to_csv("./data/test_data.txt", index=False)
            print("Data has been sliced to %60 train (train_data), %20 (validate_data) and %20 (test_data) succesfully...")
            print("Data saved into files...")

    
    def DataVisualazation(self):
        plt.figure(figsize=(10, 10))
        for label, color in zip([0, 1], ['red', 'green']):
            subset = self.train_data[self.train_data['Admitted'] == label]
            plt.scatter(subset['Exam1'], subset['Exam2'], c=color, label=f"Admitted={label}", alpha=0.7) 
        plt.title("Scatter Plot of Training Data")
        plt.xlabel("Exam 1 Scores")
        plt.ylabel("Exam 2 Scores") 
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()
        print("Data visualized successfully...")
        
    def TrainDataGet(self):
        return self.train_data
    
    def ValidateDataGet(self):
        return self.validate_data
    
    def TestDataGet(self):
        return self.test_data
    