import math
import numpy as np

"""
    Klasa reprezentująca perceptron - podstawowy model neuronu.
    
    Perceptron przetwarza dane wejściowe poprzez sumę ważoną i funkcję aktywacji,
    umożliwiając uczenie się i przewidywanie na podstawie danych.
    """
class Perceptron:
    
    """
        __init__ - Inicjalizuje nowy perceptron z domyślnymi parametrami.
        
        Ustawia funkcję tanh jako funkcję aktywacji, tworzy pustą listę wag
        i inicjalizuje bias (przesunięcie) wartością 0.
        """
    def __init__(self):
        self.activation_function = lambda x: 1.5 * math.tanh(x) + 1
        self.weights = []
        self.bias = 0


    """
        forward - Wykonuje przejście w przód (forward pass) dla perceptronu.
        
        Oblicza sumę ważoną wejść i wag, dodaje bias, a następnie
        stosuje funkcję aktywacji, aby uzyskać wyjście neuronu.
        
        Argumenty:
            x (list[float]): Wektor wejściowy zawierający dane
            
        Zwraca:
            float: Wartość wyjściowa neuronu po zastosowaniu funkcji aktywacji
        """
    def forward(self, x: list[float]) -> float:
        weighted_sum = np.dot(x, self.weights) + self.bias
        output = self.activation_function(weighted_sum)
        
        return output
    
    """
        train - Trenuje perceptron na podstawie danych wejściowych i oczekiwanych wyników.
        
        Inicjalizuje wagi i bias losowymi wartościami, a następnie iteracyjnie
        dostosowuje je na podstawie błędu między przewidywanymi a oczekiwanymi wynikami.
        
        Argumenty:
            x_train (list[list[float]]): Lista wektorów wejściowych do treningu
            y_expected (list[float]): Lista oczekiwanych wyników dla danych treningowych
            n_iter (int): Liczba iteracji (epok) treningu
            learning_rate (float): Współczynnik uczenia określający wielkość korekt wag
        """
    def train(self, x_train: list[list[float]], y_expected: list[float], n_iter: int, learning_rate: float):
        number_of_inputs = len(x_train[0])
        self.weights = np.random.randn(number_of_inputs)
        self.bias = np.random.randn()

        for _ in range(n_iter):
            for i, x in enumerate(x_train):
                y_predicted = self.forward(x)
                error = y_expected[i] - y_predicted
                correction = error * learning_rate

                self.weights += correction * x
                self.bias += correction

    """
        predict - Przewiduje wyniki dla zestawu danych wejściowych.
        
        Wykorzystuje wyuczone wagi i bias do generowania przewidywań
        dla każdego wektora wejściowego.
        
        Argumenty:
            x (list[list[float]]): Lista wektorów wejściowych do przewidywania
            
        Zwraca:
            list[float]: Lista przewidywanych wartości wyjściowych
        """
    def predict(self, x: list[list[float]]) -> list[float]:
        
        predictions = []
        for _, x in enumerate(x):
            output = self.forward(x)
            predictions.append(output) 

        return predictions