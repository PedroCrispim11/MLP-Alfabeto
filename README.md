# MLP EMNIST (Letras A-Z) em Java - Alta Performance

Implementação do zero de uma rede neural **Multi-Layer Perceptron (MLP)** escrita inteiramente em Java nativo, sem o uso de bibliotecas externas de Machine Learning. O projeto foi arquitetado com foco em **alta performance**, aplicando otimizações de baixo nível para treinar eficientemente com conjuntos de dados densos, como o **EMNIST (Letras A-Z)**.

---

## 🚀 Principais Otimizações de Desempenho

* **Zero Garbage Collection no Treino:** Arrays de ativações e deltas são pré-alocados no construtor e reutilizados iterativamente via `System.arraycopy`, eliminando pausas do coletor de lixo do Java.
* **Leitura Binária Otimizada:** Utilização de `BufferedInputStream`, `DataInputStream` e `readFully()` para carregar datasets inteiros para a memória RAM de forma quase instantânea.
* **Aceleração Matemática:** Substituição de operações custosas, como `Math.pow()`, por multiplicações diretas, além de funções otimizadas para a tangente hiperbólica (`tanh`).

---

## 📂 Pré-requisitos e Dataset (EMNIST)

Para rodar a rede neural, você precisa dos ficheiros binários do dataset **EMNIST (Letters)**. Se estiver a usar o **Ubuntu**, pode obtê-los diretamente pelo terminal:

### 1. Baixar e extrair o dataset
```bash
# Baixar o pacote oficial do NIST
wget [https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip](https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip)

# Descompactar o arquivo principal
unzip gzip.zip -d emnist_files
cd emnist_files/gzip/

# Extrair os arquivos de letras
gunzip emnist-letters-train-images-idx3-ubyte.gz
gunzip emnist-letters-train-labels-idx1-ubyte.gz
gunzip emnist-letters-test-images-idx3-ubyte.gz
gunzip emnist-letters-test-labels-idx1-ubyte.gz