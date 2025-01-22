import numpy as np

# 1. البيانات
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # المدخلات
y = np.array([[0], [1], [1], [0]])              # المخرجات

# 2. تهيئة الأوزان والإزاحة
np.random.seed(0)
w = np.random.rand(2, 1)  # الأوزان
b = np.random.rand(1)     # الإزاحة
learning_rate = 0.1       # معدل التعلم

# 3. دالة التفعيل Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4. مشتقة Sigmoid
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# 5. التدريب
epochs = 10000
for epoch in range(epochs):
    # (أ) الحساب الأمامي
    z = np.dot(X, w) + b
    a = sigmoid(z)
    
    # (ب) حساب الخسارة
    loss = -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))
    
    # (ج) الحساب العكسي
    dz = a - y
    dw = np.dot(X.T, dz) / len(X)
    db = np.sum(dz) / len(X)
    
    # (د) تحديث الأوزان
    w -= learning_rate * dw
    b -= learning_rate * db

    # طباعة الخسارة كل 1000 تكرار
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 6. التنبؤ
z = np.dot(X, w) + b
a = sigmoid(z)
print("Predictions:", a)

