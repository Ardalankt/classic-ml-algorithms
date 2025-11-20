# 📘 Classic ML & AI Algorithms (From Scratch)

This repository contains from-scratch implementations of foundational **Machine Learning** and **Artificial Intelligence** algorithms.
All code was developed two years ago as part of university assignments for the course **Computational Intelligence**.

---

## 🧠 Algorithms Included

### **Neural Learning Methods**

- **Hebbian Learning**
- **Binary Perceptron**
- **Multi-Category Perceptron**
- **Adaline (Adaptive Linear Neuron)**
- **Two-Layer Perceptron (MLP)**

### **Evolutionary / Optimization Algorithms**

- **Genetic Algorithm – Maximization Problem**
- **Genetic Algorithm – 8-Queens Problem**
- **Genetic Algorithm – Traveling Salesman Problem (TSP51)**

  - Roulette-wheel selection
  - Tournament selection

### **Logic Gate Examples**

- Hebbian AND/OR implementation
- Perceptron AND gate

---

## 🖼 Character Recognition UI (Tkinter)

The `character-xo/` directory includes a graphical interface for classifying hand-drawn **X** and **O** patterns on a 5×5 grid.

Features:

- Draw patterns using checkboxes
- Save new training examples (`trainingData.txt`)
- Retrain all implemented algorithms
- Predict using Hebb, Perceptron, Multi-category Perceptron, Adaline, or MLP

Run:

```bash
cd character-xo
python3 main.py
```
<img width="594" height="226" alt="Screenshot 2025-11-20 at 10 31 45" src="https://github.com/user-attachments/assets/4f480247-8017-48be-b04f-bffd493a5948" />

<img width="595" height="226" alt="Screenshot 2025-11-20 at 10 32 08" src="https://github.com/user-attachments/assets/5459c269-fb59-40b0-94be-23e726831cd5" />

---

## 📁 Project Structure

```
character-xo/
│   adaline.py
│   hebb.py
│   main.py
│   multiCategoryPerceptron.py
│   perceptron.py
│   twoLayerPerceptron.py
│   trainingData.txt
│
genetic algorithm(8 queen)/
│   GA(8 queen).py
│
hebb-and-or/
│   hebb-and-or.py
│
max_f(x)(genetic algorithm)/
│   GeneticAlgorithm(max).py
│
perceptron-andGate/
│   perceptron-andGate.py
│
TSP_GeneticAlgorithm/
│   TSP(GA)_roulette.py
│   TSP(GA)_tournament.py
│   TSP51.txt
```

---

## ▶️ Running Algorithms

Install requirements:

```bash
pip install numpy matplotlib
```

Run examples:

```bash
python3 perceptron-andGate/perceptron-andGate.py
python3 hebb-and-or/hebb-and-or.py
python3 "genetic algorithm(8 queen)/GA(8 queen).py"
python3 max_f(x)(genetic algorithm)/GeneticAlgorithm(max).py
python3 TSP_GeneticAlgorithm/TSP(GA)_tournament.py
```

---
