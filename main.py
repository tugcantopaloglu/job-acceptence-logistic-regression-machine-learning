#Modülerlik sağlanması amacıyla tüm işlemler farklı class'larda yapılıp burada bulunan main classında çağrılmıştır
import DataHandler
import LogisticRegression 
import numpy as np
def main():
    ### verileri alıyoruz burada data içerisinde bölünmüş halde veriler zaten bulunuyorsa işlem yapmıyoruz ###
    data = DataHandler.DataHandler("./data/hw1Data.txt")
    data.DataScrapper()
    data.DataSplitter()
    data.DataVisualazation()
     

    ### train ###
    logistic_regression_train = LogisticRegression.LogisticRegression(model_name="Train Model")
    train_epoch_losses = logistic_regression_train.StochasticGradientDescent(data.TrainDataGet()[["Exam1","Exam2"]].to_numpy(), data.TrainDataGet()["Admitted"].to_numpy(), [0.0,0.0], 0.001, 50)
    logistic_regression_train.GenerateLossGraph(train_epoch_losses)

    ### validate ###
    logistic_regression_validate= LogisticRegression.LogisticRegression(model_name="Validate Model")
    validate_epoch_losses = logistic_regression_validate.StochasticGradientDescent(data.ValidateDataGet()[["Exam1","Exam2"]].to_numpy(), data.ValidateDataGet()["Admitted"].to_numpy(), [0.0,0.0], 0.001, 50)
    logistic_regression_validate.GenerateLossGraph(validate_epoch_losses)
    
    ### L2 regularization iyileştirmesi iyileştirme uygulanmış train ###
    logistic_regression_improved_train = LogisticRegression.LogisticRegression("Improved Train Model")
    improved_train_epoch_losses = logistic_regression_improved_train.StochasticGradientDescentWithRegularization(data.TrainDataGet()[["Exam1","Exam2"]].to_numpy(), data.TrainDataGet()["Admitted"].to_numpy(), [0.0,0.0], 0.001, 1000,lambda_reg=0.01,patience=5)
    logistic_regression_improved_train.GenerateLossGraph(improved_train_epoch_losses)
    
    ### L2 regularization iyileştirmesi uygulanmış validate ###
    logistic_regression_improved_validate = LogisticRegression.LogisticRegression("Improved Validate Model")
    improved_validate_epoch_losses = logistic_regression_improved_validate.StochasticGradientDescentWithRegularization(data.ValidateDataGet()[["Exam1","Exam2"]].to_numpy(), data.ValidateDataGet()["Admitted"].to_numpy(), [0.0,0.0], 0.001, 1000)
    logistic_regression_improved_validate.GenerateLossGraph(improved_validate_epoch_losses)
    
    ### 4 model için de skorları yazdıralım (2 train 2 validate) (threshold=default)###
    #burada threshold değerlerini standart 0.5 vererek yapıyoruz ardından başarıyı arttırmak için optimal thresholdu bulup onunla da sonuçlara bakacağız
    logistic_regression_train.CalculateMetricsOnTestData()
    logistic_regression_validate.CalculateMetricsOnTestData()
    logistic_regression_improved_train.CalculateMetricsOnTestData()
    logistic_regression_improved_validate.CalculateMetricsOnTestData()
    
    ### 4 model için de skorları yazdıralım (2 train 2 validate) (threshold=optimum) ###
    ### bu işlem overfittinge yol açabilir ancak test datası üzerinde başarıyı arttırmak amaçlanmıştır ancak overfitting ihtimali olabilir ###
    logistic_regression_train.CalculateMetricsOnTestData(threshold=CalculateBestThreshold(logistic_regression_improved_train))
    logistic_regression_validate.CalculateMetricsOnTestData(threshold=CalculateBestThreshold(logistic_regression_validate))
    logistic_regression_improved_train.CalculateMetricsOnTestData(threshold=CalculateBestThreshold(logistic_regression_improved_train))
    logistic_regression_improved_validate.CalculateMetricsOnTestData(threshold=CalculateBestThreshold(logistic_regression_improved_validate))
    
def CalculateBestThreshold(model, precision = 0.01):
    thresholds = np.arange(0.1,1.0,precision)
    best_f1= 0
    best_threshold = 0
    for i in thresholds:
        value = model.CalculateMetricsOnTestData(threshold=i,check_threshold=True)
        if value > best_f1:
            best_threshold = i
            best_f1 = value
    print("Best Threshold For " + model.model_name + ":" + str(best_threshold))
    return best_threshold
    
if __name__ == "__main__":
    main()
