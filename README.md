# PyTorch LSTM Text Generation: Memory Optimization & Dynamic Quantization

This repository contains an end-to-end implementation of a text generation model using a Long Short-Term Memory (LSTM) neural network in PyTorch. The project focuses on building a functional language model and, critically, on addressing common challenges in deep learning, such as GPU memory limitations and model deployment efficiency through quantization.

## Project Description

This project demonstrates how to build, train, and deploy a basic text generation model. It learns to predict the next word in a sequence based on the context of the preceding words. The model is trained on the IMDB movie review dataset, which, while not a typical text generation dataset, serves as a practical demonstration of handling real-world text data and sequence modeling.

A significant aspect of this project is the focus on **optimizing memory usage** during training and **quantizing the model** for more efficient inference, making it suitable for deployment in resource-constrained environments.

## Features & Highlights

*   **LSTM-based Text Generation:** Implements an autoregressive language model using `nn.Embedding`, `nn.LSTM`, and `nn.Linear` layers to learn sequence patterns.
*   **Custom Data Preprocessing:** Includes robust text tokenization, vocabulary building, and dynamic sequence padding/truncation for efficient batch processing.
*   **Comprehensive GPU Memory Optimization:** Successfully tackles CUDA Out-of-Memory (OOM) errors through a combination of:
    *   Reduced batch sizes and model dimensions (`EMBED_DIM`, `HIDDEN_DIM`).
    *   Explicit garbage collection (`del`) and GPU cache clearing (`torch.cuda.empty_cache()`).
    *   Mixed-precision training (`torch.cuda.amp.autocast`, `GradScaler`) for reduced memory footprint and faster computation.
    *   Strategic environment variable configuration (`PYTORCH_CUDA_ALLOC_CONF`) to improve CUDA memory allocation.
    *   Strict sequence truncation (`MAX_SEQUENCE_LENGTH`) in data loading to limit tensor sizes.
*   **Post-Training Dynamic Quantization:** Applies `torch.quantization.quantize_dynamic` to compress the trained model, reducing its size and speeding up inference on CPU, while maintaining acceptable performance.
*   **End-to-End Pipeline:** Demonstrates a complete workflow from raw data to a trained, optimized, and inference-ready model capable of generating new text.

## Model Architecture

The core model is a sequence-to-sequence architecture:

*   **Embedding Layer (`nn.Embedding`):** Maps input token IDs to dense vector representations (`embed_dim=64`).
*   **LSTM Layer (`nn.LSTM`):** Processes the embedded sequences to capture temporal dependencies (`hidden_dim=128`, `num_layers=1`).
*   **Output Layer (`nn.Linear`):** Projects the LSTM's output back to the vocabulary space, predicting the probability distribution over the next token (`vocab_size`).

## Getting Started

### Prerequisites

*   Python 3.8+
*   `pip` package manager

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    (Generate `requirements.txt` from your environment: `pip freeze > requirements.txt`)

    *Note: If you encounter NumPy version issues, ensure `numpy<2` is specified in `requirements.txt` or install it explicitly:*
    ```bash
    pip install "numpy<2"
    ```

### Running the Code

The entire project is structured within a single Python script (or Jupyter Notebook cells if you prefer).

1.  Run the script:
    ```bash
    python your_script_name.py # Replace with your actual file name
    ```
    or execute cells sequentially in a Jupyter environment.

The script will:
*   Load and preprocess the IMDB dataset.
*   Initialize and train the LSTM model with memory optimizations.
*   Evaluate the trained model.
*   Apply dynamic quantization.
*   Demonstrate text generation using both the original and quantized models.

## Results & Observations

*(You can fill this section in after running your code)*

*   **Training Loss:** Observe how the training and validation loss decreases over epochs, indicating model learning.
*   **Memory Efficiency:** The applied optimizations successfully allowed the model to train on GPUs with limited memory, overcoming CUDA Out-of-Memory errors.
*   **Quantization Impact:** Compare the accuracy/loss of the original model versus the quantized model. Note the trade-off between performance and efficiency.
*   **Generated Text:** Examine the quality of the generated text. Early training stages might produce incoherent text, but with more epochs, the quality should improve.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
