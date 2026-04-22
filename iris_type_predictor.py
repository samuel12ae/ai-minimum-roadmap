import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from multi_layer_perceptron import MultiLayerPerceptron

"""
Skrypt do klasyfikacji gatunków irysów przy użyciu wielowarstwowego perceptronu (MLP).

Ten program wykorzystuje zbiór danych Iris do trenowania i testowania
sieci neuronowej złożonej z wielu warstw perceptronów w celu klasyfikacji
gatunków irysów na podstawie ich cech morfologicznych.

Pojedynczy perceptron nie jest w stanie osiągnąć 100% dokładności na zbiorze Iris,
ponieważ klasy versicolor i virginica nie są liniowo separowalne.
Wielowarstwowy perceptron rozwiązuje ten problem dzięki nieliniowym
funkcjom aktywacji i wielu warstwom neuronów.
"""

# Wczytujemy zbiór danych Iris, który zawiera pomiary 150 kwiatów irysów
# należących do trzech gatunków: setosa, versicolor i virginica
iris_dataset = datasets.load_iris()

# Dzielimy dane na zbiór treningowy (70%) i testowy (30%)
# x_train, x_test: cechy kwiatów (długość i szerokość płatków oraz działek kielicha)
# y_train, y_test: etykiety gatunków (0: setosa, 1: versicolor, 2: virginica)
# random_state zapewnia powtarzalność wyników przy każdym uruchomieniu
x_train, x_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.3, random_state=5)

# Ustawiamy ziarno generatora liczb losowych dla powtarzalności wyników
np.random.seed(42)

# Tworzymy wielowarstwowy perceptron o architekturze:
# - 4 neurony wejściowe (cechy kwiatów)
# - 10 neuronów w warstwie ukrytej (z funkcją aktywacji sigmoid)
# - 3 neurony wyjściowe (gatunki irysów, z funkcją aktywacji softmax)
mlp = MultiLayerPerceptron([4, 10, 3])

# Trenujemy wielowarstwowy perceptron na danych treningowych
# Parametry:
# - x_train: cechy kwiatów w zbiorze treningowym
# - y_train: odpowiadające gatunki kwiatów
# - 1000: liczba iteracji (epok) treningu
# - 0.1: współczynnik uczenia (learning rate)
mlp.train(x_train, y_train, 1000, 0.1)

# Używamy wytrenowanego wielowarstwowego perceptronu do przewidywania gatunków
# na podstawie cech kwiatów ze zbioru testowego
# MLP zwraca bezpośrednio etykiety klas dzięki funkcji softmax i argmax,
# więc nie potrzebujemy ręcznego zaokrąglania wyników
y_predicted = mlp.predict(x_test)

# Obliczamy dokładność modelu, porównując przewidywane gatunki
# z rzeczywistymi gatunkami w zbiorze testowym
accuracy = accuracy_score(y_test, y_predicted)

# Wyświetlamy dokładność modelu w procentach
print(f"Accuracy for our multi-layer perceptron: {round(accuracy, 3) * 100}%")