import math

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from perceptron import Perceptron

"""
Skrypt do klasyfikacji gatunków irysów przy użyciu perceptronu.

Ten program wykorzystuje zbiór danych Iris do trenowania i testowania
prostego perceptronu w celu klasyfikacji gatunków irysów na podstawie
ich cech morfologicznych.
"""

# Wczytujemy zbiór danych Iris, który zawiera pomiary 150 kwiatów irysów
# należących do trzech gatunków: setosa, versicolor i virginica
iris_dataset = datasets.load_iris()

# Dzielimy dane na zbiór treningowy (70%) i testowy (30%)
# x_train, x_test: cechy kwiatów (długość i szerokość płatków oraz działek kielicha)
# y_train, y_test: etykiety gatunków (0: setosa, 1: versicolor, 2: virginica)
# random_state zapewnia powtarzalność wyników przy każdym uruchomieniu
x_train, x_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.3, random_state=5)

# Tworzymy nowy obiekt perceptronu
perceptron = Perceptron()

# Trenujemy perceptron na danych treningowych
# Parametry:
# - x_train: cechy kwiatów w zbiorze treningowym
# - y_train: odpowiadające gatunki kwiatów
# - 50: liczba iteracji (epok) treningu
# - 0.02: współczynnik uczenia (learning rate)
perceptron.train(x_train, y_train, 50, 0.02)

# Używamy wytrenowanego perceptronu do przewidywania gatunków
# na podstawie cech kwiatów ze zbioru testowego
y_predicted = perceptron.predict(x_test)

# Zaokrąglamy wyniki predykcji do liczb całkowitych,
# ponieważ perceptron zwraca wartości ciągłe, a potrzebujemy etykiet klas (0, 1, 2)
y_predicted = [math.ceil(y) for y in y_predicted] 

# Obliczamy dokładność modelu, porównując przewidywane gatunki
# z rzeczywistymi gatunkami w zbiorze testowym
accuracy = accuracy_score(y_test, y_predicted)

# Wyświetlamy dokładność modelu w procentach
print(f"Accuracy for our perceptron: {round(accuracy, 3) * 100}%")