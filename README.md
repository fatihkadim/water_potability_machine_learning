# water_potability_machine_learning
# Water Potability Prediction

## 📌 Project Overview
This project aims to predict the potability of water based on its chemical properties. We use machine learning models to classify water as potable (drinkable) or not.

## 📂 Dataset
The dataset used in this project is `water_potability.csv`, which contains various water quality parameters.

## 🔧 Steps Taken
1. **Data Preprocessing**
   - Handled missing values using mean imputation.
   - Normalized features using Min-Max scaling.
2. **Exploratory Data Analysis**
   - Visualized data distributions.
   - Analyzed correlations between features.
3. **Modeling & Evaluation**
   - Trained Decision Tree and Random Forest classifiers.
   - Evaluated models using precision score and confusion matrices.
4. **Hyperparameter Optimization**
   - Used `RandomizedSearchCV` for optimizing Random Forest parameters.

## 🚀 How to Run
1. Install required libraries:  
   ```bash
   pip install pandas numpy seaborn matplotlib plotly missingno scikit-learn
   ```
2. Run the script:  
   ```bash
   python wp.py
   ```
   or open `waterquality-Copy1.ipynb` and run cells step by step.

---

# 💧 Su İçilebilirlik Tahmini

## 📌 Proje Açıklaması
Bu proje, suyun kimyasal özelliklerine göre içilebilir olup olmadığını tahmin etmeyi amaçlamaktadır. Makine öğrenmesi modelleri ile sınıflandırma yapılmaktadır.

## 📂 Veri Seti
Bu projede `water_potability.csv` veri seti kullanılmıştır ve çeşitli su kalitesi parametrelerini içermektedir.

## 🔧 Yapılan İşlemler
1. **Veri Ön İşleme**
   - Eksik veriler ortalama ile dolduruldu.
   - Özellikler Min-Max ölçeklendirme ile normalleştirildi.
2. **Keşifsel Veri Analizi**
   - Veri dağılımları görselleştirildi.
   - Özellikler arasındaki korelasyonlar incelendi.
3. **Modelleme ve Değerlendirme**
   - Karar Ağacı ve Rastgele Orman modelleri eğitildi.
   - Modeller hassasiyet skoru ve karmaşıklık matrisi ile değerlendirildi.
4. **Hiperparametre Optimizasyonu**
   - `RandomizedSearchCV` kullanılarak Rastgele Orman parametreleri optimize edildi.

## 🚀 Nasıl Çalıştırılır?
1. Gerekli kütüphaneleri yükleyin:  
   ```bash
   pip install pandas numpy seaborn matplotlib plotly missingno scikit-learn
   ```
2. Scripti çalıştırın:  
   ```bash
   python wp.py
   ```
   veya `waterquality-Copy1.ipynb` dosyasını açarak adım adım çalıştırabilirsiniz.

---

