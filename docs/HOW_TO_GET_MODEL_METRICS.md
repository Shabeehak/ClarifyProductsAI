# How to Get Model Metrics for MLflow Registration

When registering ML models in MLflow, you need to provide performance metrics. Here's how to get them professionally.

## Methods to Get Model Metrics

### Method 1: Published Benchmarks (Fastest)

For well-known models from Hugging Face, metrics are already published.

#### Where to Find Them:

**1. Hugging Face Model Card**
- Go to: https://huggingface.co/[model-name]
- Example: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
- Look for "Model Performance" or "Evaluation Results" section

**2. Research Papers**
- Original model paper (usually linked on Hugging Face)
- Example: CLIP paper → https://arxiv.org/abs/2103.00020
- Look in "Results" or "Experiments" sections

**3. Papers With Code**
- Go to: https://paperswithcode.com/
- Search for the model or dataset
- Find benchmark leaderboards

#### Common Benchmarks:

| Model Type | Benchmark Dataset | Metric |
|------------|------------------|--------|
| Image Classification | ImageNet | Accuracy |
| Sentiment Analysis | SST-2 | Accuracy |
| Summarization | CNN/DailyMail | ROUGE-L |
| OCR | Custom dataset | Character accuracy |
| NER | CoNLL-2003 | F1 Score |

### Method 2: Measure Them Yourself (Most Accurate)

Use the `measure_model_metrics.py` script to benchmark your actual deployment.

```bash
python backend/scripts/measure_model_metrics.py
```

This measures:
- **Load time:** How long to initialize the model
- **Inference time:** How long to make predictions
- **Memory usage:** RAM/GPU consumption

#### Example Output:

```json
{
  "clip": {
    "load_time_seconds": 13.75,
    "inference_time_ms": 245,
    "estimated_accuracy": 0.63
  },
  "sentiment": {
    "load_time_seconds": 0.67,
    "inference_time_ms": 72,
    "accuracy": 0.91
  }
}
```

### Method 3: Use Existing Test Results

If you've already run tests, use those results!

**Example: OCR Benchmarking**

We ran OCR tests on 29 product images:

```python
# Load results from test
results_path = "tests/data/ocr_test_dataset/ocr_comparison_results.json"
with open(results_path, 'r') as f:
    benchmark_results = json.load(f)

metrics = {
    "paddleocr_accuracy": benchmark_results["PaddleOCR"]["accuracy"] / 100,
    "paddleocr_inference_time_ms": benchmark_results["PaddleOCR"]["avg_time_ms"],
}
```

## Detailed Breakdown: Where Each Metric Came From

### CLIP Product Recognizer

```python
metrics = {
    "estimated_accuracy": 0.63,  # ← ImageNet zero-shot accuracy
    "load_time_seconds": 13.75,  # ← Measured with measure_model_metrics.py
    "inference_time_ms": 250     # ← Measured with measure_model_metrics.py
}
```

**Sources:**
- **Accuracy (0.63):** [OpenAI CLIP Paper](https://arxiv.org/abs/2103.00020), Table 10
  - ViT-B/32 on ImageNet: 63.2% zero-shot accuracy
- **Load time:** Run `measure_model_metrics.py` on your hardware
- **Inference time:** Average of 10 predictions on 224x224 images

### Sentiment Analyzer (DistilBERT)

```python
metrics = {
    "accuracy": 0.91,            # ← SST-2 benchmark
    "load_time_seconds": 0.67,   # ← Measured
    "inference_time_ms": 75      # ← Measured
}
```

**Sources:**
- **Accuracy (0.91):** Hugging Face model card
  - SST-2 dataset: 91.3% accuracy
  - Link: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
- **Load time:** Measured on CPU
- **Inference time:** Average over 10 reviews

### Summarizer (BART + Gemini)

```python
metrics = {
    "rouge_score": 0.35,              # ← CNN/DailyMail benchmark
    "load_time_seconds": 45.0,        # ← Measured
    "bart_inference_seconds": 3.0,    # ← Measured
    "gemini_inference_seconds": 0.8,  # ← Measured
}
```

**Sources:**
- **ROUGE score (0.35):** [BART Paper](https://arxiv.org/abs/1910.13461), Table 1
  - CNN/DailyMail dataset: 35.0 ROUGE-L
- **Load time:** Measured (BART is 1.6 GB, takes time to load)
- **BART inference:** Average of 5 summaries on multi-review text
- **Gemini inference:** Measured with Gemini API calls

### OCR (PaddleOCR + EasyOCR)

```python
metrics = {
    "paddleocr_accuracy": 0.4828,     # ← Our test dataset
    "paddleocr_inference_time_ms": 6934,  # ← Our test dataset
    "easyocr_accuracy": 0.4643,       # ← Our test dataset
}
```

**Sources:**
- **All metrics:** `tests/data/ocr_test_dataset/ocr_comparison_results.json`
- **Test dataset:** 29 product images we collected
- **Methodology:** Ran both OCR engines, compared to ground truth

## Best Practices

### 1. **Combine Sources**

```python
metrics = {
    "accuracy": 0.91,           # From benchmark (SST-2)
    "load_time_seconds": 0.67,  # Measured on your hardware
    "inference_time_ms": 75,    # Measured on your hardware
}
```

- Use **published benchmarks** for accuracy/quality metrics
- Use **measurements** for performance (time, memory)

### 2. **Document Your Sources**

```python
# In register_models.py
metrics = {
    "estimated_accuracy": 0.63,  # ImageNet zero-shot accuracy (OpenAI CLIP paper)
    "load_time_seconds": 13.75,  # Measured on Intel i7 CPU
    "inference_time_ms": 250     # Average of 10 runs
}
```

### 3. **Update When You Change Hardware**

If you move from local → cloud, re-measure:

```bash
# On cloud server
python scripts/measure_model_metrics.py
```

### 4. **Use Realistic Test Data**

Don't test with dummy data! Use:
- Real product images (not random noise)
- Actual review text (not "test test test")
- Representative dataset size (at least 10-20 samples)

## Interview-Ready Explanation

When asked "How did you get these metrics?":

> "I used a combination of published benchmarks and real-world measurements. For accuracy metrics like the 91% for sentiment analysis, I referenced the official SST-2 benchmark from the Hugging Face model card. For performance metrics like load time and inference time, I created a benchmarking script that measures actual performance on our deployment hardware. This gives us both theoretical accuracy (from research) and practical performance (from our environment)."

## Common Mistakes to Avoid

❌ **Using made-up numbers**
```python
metrics = {
    "accuracy": 0.99,  # ← Too good to be true!
}
```

❌ **Not updating after hardware changes**
```python
# Measured on laptop, but deploying to cloud
"load_time_seconds": 2.0  # ← Will be different in production!
```

❌ **Forgetting to document sources**
```python
"accuracy": 0.91  # ← Where did this come from?
```

✅ **Professional approach**
```python
metrics = {
    "accuracy": 0.91,  # SST-2 benchmark (Hugging Face)
    "load_time_seconds": 0.67,  # Measured on deployment server
    "inference_time_ms": 75  # Average of 100 predictions
}

# Comment explaining measurement methodology
# Measured using scripts/measure_model_metrics.py
# Test dataset: 100 real product reviews
# Hardware: AWS t3.medium instance
# Date: 2025-11-01
```

## Quick Reference: Where to Look

| Metric | Where to Find It |
|--------|------------------|
| Accuracy | Hugging Face model card → "Evaluation Results" |
| ROUGE/BLEU | Original paper → Results section |
| Load time | Run `measure_model_metrics.py` |
| Inference time | Run `measure_model_metrics.py` |
| F1 Score | Papers With Code leaderboard |
| Memory usage | `torch.cuda.memory_allocated()` or `psutil` |

## Tools You Can Use

1. **`measure_model_metrics.py`** - Custom benchmarking script (this project)
2. **`transformers-cli`** - Hugging Face benchmark tool
3. **MLflow** - Track metrics automatically during experiments
4. **Weights & Biases** - Advanced experiment tracking
5. **TensorBoard** - Visualize training metrics

---

**Remember:** It's better to have measured metrics from a smaller dataset than to guess!
