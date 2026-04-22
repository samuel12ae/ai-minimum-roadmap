import numpy as np

"""
    Klasa reprezentująca wielowarstwowy perceptron (MLP) - sieć neuronowa
    składająca się z wielu warstw perceptronów.
    
    MLP przetwarza dane wejściowe przez wiele warstw neuronów z nieliniowymi
    funkcjami aktywacji, umożliwiając rozwiązywanie problemów nieliniowo
    separowalnych, takich jak klasyfikacja wielu klas w zbiorze Iris.
"""
class MultiLayerPerceptron:

    """
        __init__ - Inicjalizuje wielowarstwowy perceptron z podaną architekturą.
        
        Tworzy strukturę sieci neuronowej o zadanych rozmiarach warstw.
        Wagi i biasy są inicjalizowane podczas treningu.
        
        Argumenty:
            layer_sizes (list[int]): Lista rozmiarów kolejnych warstw sieci,
                np. [4, 10, 3] oznacza 4 wejścia, 10 neuronów w warstwie
                ukrytej i 3 neurony wyjściowe
    """
    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

    """
        _sigmoid - Funkcja aktywacji sigmoid.
        
        Przekształca wartość wejściową na zakres (0, 1), co umożliwia
        modelowanie nieliniowych zależności w warstwach ukrytych.
        
        Argumenty:
            x (numpy.ndarray): Wartość wejściowa
            
        Zwraca:
            numpy.ndarray: Wartość po zastosowaniu funkcji sigmoid
    """
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    """
        _sigmoid_derivative - Pochodna funkcji sigmoid.
        
        Oblicza pochodną funkcji sigmoid na podstawie wartości
        przed aktywacją. Używana podczas propagacji wstecznej
        do obliczania gradientów.
        
        Argumenty:
            x (numpy.ndarray): Wartość przed aktywacją (pre-activation)
            
        Zwraca:
            numpy.ndarray: Wartość pochodnej funkcji sigmoid
    """
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        s = self._sigmoid(x)
        return s * (1 - s)

    """
        _softmax - Funkcja aktywacji softmax dla warstwy wyjściowej.
        
        Przekształca wektor wartości na rozkład prawdopodobieństwa,
        gdzie suma wszystkich elementów wynosi 1. Używana w warstwie
        wyjściowej do klasyfikacji wieloklasowej.
        
        Argumenty:
            x (numpy.ndarray): Wektor wartości wejściowych
            
        Zwraca:
            numpy.ndarray: Wektor prawdopodobieństw dla każdej klasy
    """
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    """
        _one_hot - Kodowanie one-hot etykiety klasy.
        
        Przekształca etykietę klasy (np. 2) na wektor binarny
        (np. [0, 0, 1]), co jest wymagane do obliczania błędu
        w warstwie wyjściowej z funkcją softmax.
        
        Argumenty:
            y (int): Etykieta klasy (numer klasy)
            num_classes (int): Całkowita liczba klas
            
        Zwraca:
            numpy.ndarray: Wektor one-hot o długości num_classes
    """
    def _one_hot(self, y: int, num_classes: int) -> np.ndarray:
        one_hot = np.zeros(num_classes)
        one_hot[int(y)] = 1.0
        return one_hot

    """
        forward - Wykonuje przejście w przód (forward pass) przez całą sieć.
        
        Dane wejściowe przechodzą kolejno przez każdą warstwę sieci.
        Warstwy ukryte używają funkcji sigmoid, a warstwa wyjściowa
        używa funkcji softmax. Wartości pośrednie są zapamiętywane
        na potrzeby propagacji wstecznej.
        
        Argumenty:
            x (numpy.ndarray): Wektor wejściowy zawierający dane
            
        Zwraca:
            numpy.ndarray: Wektor prawdopodobieństw wyjściowych dla każdej klasy
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z_values = []
        self.a_values = [np.array(x)]

        for i in range(len(self.weights)):
            z = np.dot(self.a_values[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if i == len(self.weights) - 1:
                # Warstwa wyjściowa - softmax
                a = self._softmax(z)
            else:
                # Warstwy ukryte - sigmoid
                a = self._sigmoid(z)

            self.a_values.append(a)

        return self.a_values[-1]

    """
        train - Trenuje wielowarstwowy perceptron metodą propagacji wstecznej.
        
        Inicjalizuje wagi metodą Xaviera, a następnie dla każdej epoki
        i każdej próbki treningowej wykonuje przejście w przód, oblicza
        błąd, propaguje go wstecz przez warstwy i aktualizuje wagi
        oraz biasy zgodnie z obliczonymi gradientami.
        
        Argumenty:
            x_train (list[list[float]]): Lista wektorów wejściowych do treningu
            y_train (list[int]): Lista etykiet klas dla danych treningowych
            n_iter (int): Liczba iteracji (epok) treningu
            learning_rate (float): Współczynnik uczenia określający wielkość korekt wag
    """
    def train(self, x_train: list[list[float]], y_train: list[int], n_iter: int, learning_rate: float):
        # Inicjalizacja wag metodą Xaviera dla lepszej zbieżności
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            limit = np.sqrt(6 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit, (self.layer_sizes[i], self.layer_sizes[i + 1]))
            b = np.zeros(self.layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

        num_classes = self.layer_sizes[-1]

        for _ in range(n_iter):
            for i in range(len(x_train)):
                x = np.array(x_train[i])
                target = self._one_hot(y_train[i], num_classes)

                # Przejście w przód
                output = self.forward(x)

                # Propagacja wsteczna - obliczanie błędów (delt)
                # Błąd warstwy wyjściowej (softmax + entropia krzyżowa)
                deltas = [output - target]

                # Błędy warstw ukrytych
                for j in range(len(self.weights) - 2, -1, -1):
                    error = np.dot(deltas[0], self.weights[j + 1].T)
                    delta = error * self._sigmoid_derivative(self.z_values[j])
                    deltas.insert(0, delta)

                # Aktualizacja wag i biasów na podstawie gradientów
                for j in range(len(self.weights)):
                    self.weights[j] -= learning_rate * np.outer(self.a_values[j], deltas[j])
                    self.biases[j] -= learning_rate * deltas[j]

    """
        predict - Przewiduje klasy dla zestawu danych wejściowych.
        
        Dla każdej próbki wykonuje przejście w przód przez sieć
        i wybiera klasę o najwyższym prawdopodobieństwie (argmax).
        
        Argumenty:
            x (list[list[float]]): Lista wektorów wejściowych do przewidywania
            
        Zwraca:
            list[int]: Lista przewidywanych etykiet klas
    """
    def predict(self, x: list[list[float]]) -> list[int]:
        predictions = []
        for sample in x:
            output = self.forward(np.array(sample))
            predictions.append(np.argmax(output))

        return predictions
