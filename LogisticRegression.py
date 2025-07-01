#burada her şeyi kendim yazmaya çalıştım hesaplamaları derste öğrendiğimiz fonksiyonlardan aldığım notlardan oluşturdum aşağıdaki kütüphaneleri sadece matematik işlemleri ve görselleştirme için ekledim
import math
import matplotlib.pyplot as plt
import pandas as pd
import DataHandler
class LogisticRegression():
    
    def __init__(self,model_name) -> None:
        self.model_name = model_name
    
    #x e^-x si ifade ediyor
    def Sigmoid(self, x):
        e = 2.718281828459
        return 1/(1 + e**(-1*x))
    
    #gerçek değer ile bulunan değeri alıyoruz ve yine cross entropy'de kullanıyoruz bunu yapark 1 ile 1-epsilon arasına sıkıştırıyoruz
    def CrossEntropyLoss(self, y_target, y_predict):
        epsilon = 1e-15
        y_predict = max(min(y_predict,1 - epsilon), epsilon)
        return -(y_target * math.log(y_predict) + (1 - y_target) * math.log(1 - y_predict))
        


    #x girdi özellikleri ve y çıkış etiketleri olacak şekilde weights güncellenmiş ağırlıklardır öğrenme oranı ve epoch sayısını fonksiyonda alıyoruz bu şekilde farklı epochlarda dönebiliriz. her örnek için bu işlemleri yapıyoruz ve kendi oluşturduğumuz fonksiyonlar ile yapıyoruz bu işlemi yaparken her ağırlık için ağırlar gradientini de buluyoruz buradaki epoch loss her epochtaki bu loss'u gösteriyor 
    def StochasticGradientDescent(self, x, y, weights, learning_rate, epochs):
        sample_num = len(y)
        epoch_losses = []
        with open("./results/epoch_loss_output.txt", 'a+') as f:
            f.write("Epoch started for not improvized SGD:" + "\n")
        for epoch in range(epochs):
            epoch_loss = 0 # her epoch başında sıfırlıyorum ki doğru sonucu bulabilelim burada daha önce sıfırlamadığımdan yanlış grafikler elde ettim
            for i in range(sample_num):
                lineer_combination = sum(weights[j] * x[i][j] for j in range(len(weights)))
                y_predict = self.Sigmoid(lineer_combination)
                ### loss hesaplıyoruz ###
                loss = self.CrossEntropyLoss(y[i], y_predict)
                epoch_loss += loss
                ### loss hesaplıyoruz ###
                gradients = [(y_predict - y[i]) * x[i][j] for j in range(len(weights))]
                weights = [weights[j] - learning_rate * gradients[j] for j in range(len(weights))]
            epoch_losses.append(epoch_loss / sample_num)
            with open("./results/epoch_loss_output.txt", 'a+') as f:
                f.write("Epoch Loss on " + str(epoch) + ": " + str(epoch_loss) + "\n")
        with open("./results/weights.txt", 'a+') as f:
            f.write("Model Name: " + self.model_name + "\n")
            f.write("Weights: " + str(weights) + "\n")
        self.weights = weights
        return epoch_losses
    
    def GenerateLossGraph(self, epoch_losses):
        plt.figure(figsize=(10, 10))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label=self.model_name+" Loss")
        plt.title("Cross-Entropy Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Average Cross-Entropy Loss")
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()
    
    #grafiği manuel inceleyip nerede durduracağımıza karar vermenin yanında aynı işi yapacak bir fonksiyon geliştirdim önce her epochtaki lose farkları ile bir data frame oluşturuyoruz ardından bizim belirlediğimiz değere gelip gelmediğini kontrol ediyoruz TEKRARDAN REGULARIZATION BULUNAN İÇİNE İMPLEMENTE EDİLDİ BUNDAN DOLAYI KULLANILMIYOR
    def CalculateStoppingEpoch(self, stop_point):
        loss_differences = [abs(self.epoch_losses[i] - self.epoch_losses[i - 1]) for i in range(1, len(self.epoch_losses))]
        epoch_analysis = pd.DataFrame({
            "Epoch": list(range(2, len(self.epoch_losses) + 1)),
            "Loss Difference": loss_differences
        })
        stopping_epoch = epoch_analysis[epoch_analysis["Loss Difference"] < stop_point].iloc[0]["Epoch"] if not epoch_analysis[epoch_analysis["Loss Difference"] < stop_point].empty else None
        
        return stopping_epoch
    
    #daha önce yaptığımız işleme L2 regularization ekledik burada lambda_reg L2 regularization katsayısı ve patience ise early stopping için aynı değerin gelebileceği maksimum patience yani aslında burada hem calculate stopping epoch'ta yapmaya çalıştığım şeyi birleştirdim hem de regularization ekleyerek sonuçların overfittingten kurtulmasını sağlamaya çalıştım. kısaca loss hesapladığımız bölümde regularization da ekleyerek loss hesapladık
    def StochasticGradientDescentWithRegularization(self, x, y, weights, learning_rate, epochs, lambda_reg=0.01, patience=5):
        sample_num = len(y)
        epoch_losses = []
        best_loss = float('inf')
        patience_counter = 0
        stopping_epoch = None
        with open("./results/epoch_loss_output.txt", 'a+') as f:
            f.write("Improvized Model: " + self.model_name + " epoch started..." + "\n")
        for epoch in range(epochs):
            epoch_loss = 0

            for i in range(sample_num):
                lineer_combination = sum(weights[j] * x[i][j] for j in range(len(weights)))
                y_prediction = self.Sigmoid(lineer_combination)
                loss = self.CrossEntropyLoss(y[i], y_prediction) + (lambda_reg / 2) * sum(w**2 for w in weights)
                epoch_loss += loss
                gradients = [(y_prediction - y[i]) * x[i][j] + lambda_reg * weights[j]for j in range(len(weights))]
                weights = [weights[j] - learning_rate * gradients[j] for j in range(len(weights))]
            avg_epoch_loss = epoch_loss / sample_num
            with open("./results/epoch_loss_output.txt", 'a+') as f:
                f.write("Epoch Loss on " + str(epoch) + ": " + str(epoch_loss) + "\n")
                f.write("Average Epoch Loss on " + str(epoch) + ": " + str(avg_epoch_loss) + "\n")
            epoch_losses.append(avg_epoch_loss)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                stopping_epoch = epoch + 1
                print(f"Early stopping has been seen at epoch {stopping_epoch}")
                break
            with open("./results/epoch_loss_output.txt", 'a+') as f:
                f.write("Model Name: " + self.model_name + "\n")
                f.write("Average Epoch Loss on " + str(epoch) + ": " + str(avg_epoch_loss) + "\n")
        with open("./results/weights.txt", 'a+') as f:
            f.write("Model Name: " + self.model_name + "\n")
            f.write("Weights: " + str(weights) + "\n")
        self.weights = weights
        return epoch_losses
    
    #formüllere göre metrikleri hesaplıyoruz: TP = Doğru Pozitif (doğru tahmin ettik ve doğru olmalıydı), FP = yanlış pozitif (doğru tahmin ettik yanlış olmalıydı), TN = doğru negatif (yanlış tahmin ettik yanlış olmalıydı), FN = yanlış negatif (yanlış tahmin ettik ama doğru olmalıydı)
    #accuracy = doğru tahmin sayısı / toplam tahmin sayısı
    #kesinlik = TP/(TP+FP)
    #duyarlılık = TP/(TP+FP)
    #F1 skoru = 2*((kesinlik*duyarlılık)/(kesinlik+duyarlılık))
    def Metrics(self, y_target, y_predicted):
        TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_target, y_predicted))
        FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_target, y_predicted))
        TN = sum((yt == 0 and yp == 0) for yt, yp in zip(y_target, y_predicted))
        FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_target, y_predicted))

        accuracy = (TP + TN) / len(y_target) if len(y_target) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return accuracy, precision, recall, f1_score
    
    #test ile deneme yaparak skorları hesaplıyoruz
    def CalculateMetricsOnTestData(self,threshold = 0.5,check_threshold=False):
        data_handler = DataHandler.DataHandler("./data/hw1Data.txt")
        data_handler.DataSplitter()
        x_test = data_handler.TestDataGet()[["Exam1", "Exam2"]].to_numpy()
        y_test = data_handler.TestDataGet()["Admitted"].to_numpy()
        y_prediction = []
        for i in range(len(y_test)):
            lineer_combination = sum(self.weights[j] * x_test[i][j] for j in range(len(self.weights)))
            y_prediction.append(1 if self.Sigmoid(lineer_combination) >= threshold else 0)

        # Metrikleri hesapla
        accuracy, precision, recall, f1_score = self.Metrics(y_test, y_prediction)
        
        #optimum thresholdu bulurken bu çıktıları vermesini istemiyoruz bundan dolayı böyle bir kontrol yapıyoruz
        if (check_threshold == False):
            if threshold != 0.5:
                print("MODEL THRESHOLD CHANGED TO OPTIMUM CALCULATING AGAIN")
            print("")
            print("###")
            print(self.model_name + " Scores:")
            print("Model Evaluation on Test Set:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1_score:.4f}")
            print("###")
            print("")
            print("Percentage Calculation:")
            print(f"Accuracy: %{accuracy*100:.2f}")
            print(f"Precision: %{precision*100:.2f}")
            print(f"Recall: %{recall*100:.2f}")
            print(f"F1-Score: %{f1_score*100:.2f}")
            
            with open("./results/scores.txt", 'a+') as f:
                if threshold != 0.5:
                    f.write("MODEL THRESHOLD CHANGED TO OPTIMUM CALCULATING AGAIN\n")
                f.write("###\n")
                f.write(self.model_name + " Scores:\n")
                f.write("Model Evaluation on Test Set:\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1-Score: {f1_score:.4f}\n")
                f.write("###\n")
                f.write(f"Accuracy: %{accuracy*100:.2f}\n")
                f.write(f"Precision: %{precision*100:.2f}\n")
                f.write(f"Recall: %{recall*100:.2f}\n")
                f.write(f"F1-Score: %{f1_score*100:.2f}\n")
                f.write("###\n")
            
        
        return f1_score
    

        


