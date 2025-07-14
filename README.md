# autograd-zero
A tiny, easy-to-understand automatic differentiation engine for scalar values..
---

##  What’s Working Right Now

This project is a minimal clone of what powers tools like PyTorch — but built from scratch for learning and fun.

### ✅ Currently Implemented:

- 🧩 **Core Neuron class**  
  Tracks value, gradient, and the operation used to create each node.

- ➕ **Basic math operations**  
  You can do `+`, `-`, `*`, and even `**` (power) between Neurons or numbers.

- 🔁 **Backpropagation (Autodiff)**  
  Computes gradients using reverse-mode autodiff with a topological sort.

- 📊 **Graph Visualization**  
  Generate computation graphs using **Graphviz** for a better understanding of dependencies.

---

## 🛠️ What’s Coming Next (v2 Ideas)

- 🔢 More operations: `/`, `//`, `%`, etc.  
- 🔥 Activation functions: `relu`, `tanh`, `sigmoid`, `lrelu`, and more  
- 🎯 Loss functions like MSE,
- 🧹 `zero_grad()` utility  
- 🧮 Vector and matrix support for batched computation

---

## 📚 Why am I doing this?

To truly understand backpropagation and how automatic differentiation and DAG works under the hood.
Learning Topological sort is cherry on top.

---

## 📄 License

MIT — Free to use, share, and modify. Built for learning.
