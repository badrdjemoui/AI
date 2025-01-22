import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# البيانات
X = np.array([[7, 150], [6, 140], [5, 130], [9, 200], [8, 190], [10, 210]])  # الميزات
y = np.array(["Apple", "Apple", "Apple", "Orange", "Orange", "Orange"])  # التصنيفات

# الفاكهة الجديدة
new_fruit = np.array([[8, 180]])

# إنشاء نموذج KNN
knn = KNeighborsClassifier(n_neighbors=3)  # نختار 3 جيران
knn.fit(X, y)  # تدريب النموذج

# التنبؤ
prediction = knn.predict(new_fruit)
print(f"The new fruit is classified as: {prediction[0]}")
