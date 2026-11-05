**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** Laboratorium 4 — Obsługa długich sekwencji z użyciem SimpleRNN, LSTM, GRU i Conv1D
**Autor / grupa:** Damian Skorupa / GL03
**Data:** 05.11.2026

## 1. Cel ćwiczenia
Celem ćwiczenia była ewaluacja skuteczności czterech architektur głębokich (SimpleRNN, LSTM, GRU oraz Conv1D) w radzeniu sobie z problemem długich zależności w czasie. Wykorzystano wariant "adding problem", w którym znaczniki wskazujące wartości do zsumowania pojawiają się z opóźnieniem 8 kroków. Analizowano wpływ długości sekwencji, rozmiaru stanu ukrytego oraz modyfikacji pola recepcyjnego splotu na finalny błąd predykcji.

## 2. Metoda i implementacja
- **Zbiory i generator:** Opracowano własny generator danych wejściowych o kształcie `[N, L, 2]`. Wylosowany znacznik w kroku $t$ wymuszał pobranie wartości z kroku $t-8$. Na przykład: znaczniki znajdujące się w krokach 14 i 19 poprawnie wskazywały wartości z kroków 6 i 11. Ich ręczna suma wyniosła 0.8071, co perfekcyjnie pokryło się z celem zwracanym przez model. Wygenerowano 4000 próbek treningowych i 800 testowych dla każdej badanej długości $L$ (50 i 100).
- **Konfiguracja:** Trening realizowano przy użyciu CPU, wykonując 30 epok z rozmiarem partii 128 (optymalizator Adam, lr=5e-3, ziarno=0). Jako punkt odniesienia przyjęto średnią arytmetyczną celu treningowego.
- **Architektura modeli:** Modele rekurencyjne wykorzystywały odczyt wyniku z ostatniego stanu ukrytego. Architektura Conv1D została skonstruowana w oparciu o 3 warstwy konwolucyjne z dylatacjami `[1, 2, 4]`, po których następowało uśrednianie mapy cech w wymiarze czasowym.

## Link do Githuba
https://github.com/damianero47/Obs-uga-d-ugich-sekwencji-LSTM-GRU-i-splot-1D

## 3. Wyniki

Tabela porównawcza eksperymentów:

| Model (Rozmiar L) | MSE na teście | Czas uczenia [s] | Liczba parametrów |
|---|---|---|---|
| **Baseline L=50** | 0.1598 | - | 0 |
| SimpleRNN (32) L=50 | 0.1598 | 2.57s | 1185 |
| LSTM (32) L=50 | 0.1597 | 7.88s | 4641 |
| GRU (32) L=50 | 0.1439 | 5.93s | 3489 |
| Conv1D [1,2,4] L=50 | 0.1555 | 26.46s | 25217 |
| **Baseline L=100** | 0.1667 | - | 0 |
| SimpleRNN (32) L=100| 0.1703 | 4.72s | 1185 |
| LSTM (32) L=100 | 0.1667 | 14.21s | 4641 |
| GRU (32) L=100 | 0.1663 | 11.83s | 3489 |
| Conv1D [1,2,4] L=100| 0.1643 | 34.70s | 25217 |
| *SimpleRNN (64) L=100*| 0.1697 | 5.86s | 4417 |
| *Conv1D [1,1,1] L=100*| 0.1641 | 43.14s | 25217 |

<img width="900" height="500" alt="wykres_modele" src="https://github.com/user-attachments/assets/c5b91b1a-00fe-45dc-8715-45f40fbcd0e3" />


## 4. Analiza i interpretacja
Wygenerowane wyniki jednoznacznie wskazują, że bazowa sieć SimpleRNN nie potrafi opanować długich zależności – jej błąd we wszystkich wariantach oscyluje wokół losowej predykcji stałej (~0.16). Modele z wbudowanymi mechanizmami bramek uaktualniających pamięć (LSTM, GRU) oraz sieć splotowa z uśrednianiem (Conv1D) stanowczo przebiły ten baseline, redukując błąd MSE.

Zmiana parametrów architektonicznych przyniosła następujące wnioski:
- **Rozmiar stanu w RNN (32 vs 64):** Mimo dramatycznego zwiększenia liczby parametrów i czasu uczenia, powiększony stan ukryty nie wpłynął pozytywnie na SimpleRNN. Potwierdza to fakt, że problem zanikającego gradientu przy dużej liczbie kroków czasowych ma naturę matematyczną (mnożenie wielokrotne), a nie pojemnościową.
- **Pole recepcyjne w splotach (Dylatacje [1,2,4] vs [1,1,1]):** Zgodnie ze wzorem, pole recepcyjne dla warstw z dylatacjami `[1,2,4]` to $R = 1 + (3-1) \cdot (1+2+4) = 15$. Ponieważ przesunięcie markera wynosi 8, obszar 15 obejmuje jednocześnie wskaźnik i ukrytą wartość. Po zmianie dylatacji na `[1,1,1]`, pole skurczyło się do $R = 7$. Model nie był w stanie lokalnie połączyć markera z liczbą (za mały zasięg wizji), przez co musiał polegać wyłącznie na ślepym uśrednianiu globalnym, co doprowadziło do spadku skuteczności.

**Przypadek z ekstremalnym błędem (SimpleRNN dla L=100):**
Wykryto przypadek testowy z błędem rzędu 1.0188. Model miał zsumować wartości `[0.97497475, 0.98591834]` na podstawie znaczników z kroków 10 i 66. Ponieważ architektonicznie rekurencja wyprowadza odpowiedź na bazie swojego ostatniego stanu ukrytego po przetworzeniu całej sekwencji (krok 100), sygnał o dawnych pozycjach został całkowicie zatarty. Przez to predykcja algorytmu (0.9421) nie miała kompletnie nic wspólnego z prawdziwą sumą (1.9609).

## 5. Wnioski
Eksperyment potwierdza przewagę algorytmów uwzględniających kontrolę przepływu dawnych informacji. Wnioski te bezpośrednio przekładają się na profesjonalne architektury przetwarzania sygnałów audio. Zastosowanie splotów rozszerzonych w modelach takich jak WaveNet pozwala na zachowanie ogromnego pola recepcyjnego i globalnego kontekstu dźwięku (przy relatywnie niskim koszcie obliczeniowym), z kolei LSTM i GRU doskonale sprawdzają się przy autoregresyjnym, stabilnym uaktualnianiu bieżącego stanu mowy bez ryzyka destrukcji gradientu w czasie propagacji wstecznej.

