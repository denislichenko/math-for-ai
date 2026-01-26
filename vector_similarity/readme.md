# 🧠 Project 1 — Vector Similarity Engine

**(Linear Algebra + Embeddings Mindset)**

This project is a small but fundamental implementation of a **vector similarity engine**. It demonstrates how modern Machine Learning systems compare meanings using pure linear algebra.

The goal is not performance, but **deep understanding** of why embeddings, cosine similarity, and vector databases work.

---

## 🎯 Goal of the Project

The main objective of this project is to understand:

* What vectors represent in Machine Learning
* How dot product works
* What vector norms (L2 norm) are
* How cosine similarity is computed
* Why embeddings-based systems work (semantic search, RAG, recommender systems)

This project intentionally starts **without NumPy**, and only later introduces it, to avoid hiding the math behind libraries.

---

## 📌 What Was Implemented

1. Manual creation of vectors
2. From-scratch implementation (pure Python):

   * Dot product
   * L2 norm
   * Vector normalization
   * Cosine similarity
3. The same logic rewritten using NumPy
4. A simple similarity search: finding the most similar vector to a query

---

## 🧪 Example Task

We represent words as vectors (toy example):

```python
texts = {
    "apple": [1, 0, 0],
    "banana": [0.9, 0.1, 0],
    "car": [0, 0, 1]
}

print(get_most_similar_vector("apple", texts))
```

**Expected output:**

```text
banana
```

Even in this simplified setup, the system correctly identifies that **"apple"** is closer to **"banana"** than to **"car"**.

---

## 📐 Math Behind the Project

> **Note:** GitHub README does not render LaTeX math by default, so formulas are written in plain text.

---

### 1️⃣ Vectors

A vector is an ordered list of numbers:

```
v = (v1, v2, ..., vn)
```

In ML, vectors are used to represent:

* words
* sentences
* images
* users
* products

---

### 2️⃣ Dot Product (Scalar Product)

For two vectors:

```
a = (a1, a2, ..., an)
b = (b1, b2, ..., bn)
```

The dot product is defined as:

```
a · b = a1*b1 + a2*b2 + ... + an*bn
```

Geometric meaning:

```
a · b = |a| * |b| * cos(theta)
```

Where `theta` is the angle between vectors.

* Large positive value → vectors point in a similar direction
* Zero → vectors are orthogonal (unrelated)
* Negative → vectors point in opposite directions

---

### 3️⃣ L2 Norm (Vector Length)

The L2 norm measures the length of a vector:

```
||v|| = sqrt(v1^2 + v2^2 + ... + vn^2)
```

In ML, the norm represents magnitude, which is often unwanted when comparing meanings.

---

### 4️⃣ Vector Normalization

To remove magnitude influence, vectors are normalized:

```
v_normalized = v / ||v||
```

After normalization, all vectors lie on the unit hypersphere.

---

### 5️⃣ Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors:

```
cosine_similarity(a, b) = (a · b) / (||a|| * ||b||)
```

If vectors are already normalized:

```
cosine_similarity(a, b) = a · b
```

Interpretation:

| Value | Meaning             |
| ----- | ------------------- |
| 1.0   | Identical direction |
| ~0.8  | Very similar        |
| 0.0   | Unrelated           |
| < 0   | Opposite meanings   |

---

### 2️⃣ Dot Product (Scalar Product)

For two vectors:

[
\vec{a} = (a_1, a_2, ..., a_n), \quad
\vec{b} = (b_1, b_2, ..., b_n)
]

Where (\theta) is the angle between vectors.

* Large positive value → vectors point in a similar direction
* Zero → vectors are orthogonal (unrelated)
* Negative → vectors point in opposite directions

---

### 3️⃣ L2 Norm (Vector Length)

The L2 norm measures the length of a vector. In ML, the norm represents **magnitude**, which is often unwanted when comparing meanings.

---

### 4️⃣ Vector Normalization

To remove magnitude influence, vectors are normalized.
After normalization, all vectors lie on the **unit hypersphere**.

---

### 5️⃣ Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors.

**Interpretation:**

| Value | Meaning             |
| ----- | ------------------- |
| 1.0   | Identical direction |
| ~0.8  | Very similar        |
| 0.0   | Unrelated           |
| < 0   | Opposite meanings   |

---

## 🧠 Why This Is Important in Machine Learning

This exact mechanism is used in:

* Semantic search
* Retrieval-Augmented Generation (RAG)
* Recommendation systems
* Clustering
* Document ranking
* Vector databases (FAISS, Pinecone, Milvus)

In practice:

* Documents → embeddings (vectors)
* Query → embedding
* Search → cosine similarity between vectors

This project reproduces the **core logic** of those systems in a minimal, transparent way.

---

## 🔍 Example Interpretation

Using the example:

```text
apple  = [1,   0,   0]
banana = [0.9, 0.1, 0]
car    = [0,   0,   1]
```

* `apple` and `banana` point in almost the same direction
* `car` is orthogonal to both

Cosine similarity captures this relationship geometrically.

---

## 🚀 Key Takeaways

* Embeddings are **geometry**, not magic
* Similarity search is just vector math
* Cosine similarity compares directions, not magnitudes
* L2 normalization is critical
* Vector databases scale this exact idea

---

## 📦 Possible Extensions

* Top-K similarity search
* Batch similarity using matrix multiplication
* Visualization of vectors
* Integration with real embeddings (OpenAI, Sentence Transformers)
* FAISS-based nearest neighbor search

---

## ✅ Conclusion

This project builds a strong mental model for understanding modern ML systems. By implementing vector similarity from scratch, it becomes clear how embeddings, semantic search, and RAG pipelines work under the hood.

**If you understand this project, you understand the foundation of modern AI systems.**
