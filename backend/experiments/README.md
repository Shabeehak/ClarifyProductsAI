# Experimental Approaches Archive

This directory contains experimental approaches tested during the development of the product recognition system. These files are kept for documentation purposes to show the iterative development process.

## Experiments Conducted

### 1. NER-Based Extraction (FAILED - 0% Accuracy)
**Files:**
- `services/ner_service.py`
- `tests/test_ner_improvement.py`

**Approach:** Used GLiNER (zero-shot NER model) to extract product names from OCR text.

**Result:** 0/8 correct (0% accuracy)

**Why it failed:**
- Only extracted brand names, not full product names
- Couldn't handle product descriptions
- Not suitable for unstructured OCR output

**Lesson Learned:** NER models need structured text, not raw OCR output with marketing text.

---

### 2. Rule-Based Text Cleaning (LIMITED - 25% Accuracy)
**Files:**
- `services/text_cleaner_service.py`
- `tests/test_text_cleaning.py`

**Approach:** Hardcoded patterns to clean OCR text and correct common mistakes.

**Result:** 2/8 correct (25% accuracy)

**Why it failed:**
- Hardcoded corrections only work for known products
- Cannot scale to unknown products
- Requires manual maintenance of pattern dictionary

**Lesson Learned:** Rule-based approaches don't generalize to unseen products.

---

### 3. Gemini OCR-Only (FAILED - 12.5% Accuracy + Hallucination)
**Files:**
- `services/gemini_product_extractor.py`
- `tests/test_gemini_extraction.py`

**Approach:** Send only OCR text to Gemini AI for product name extraction.

**Result:** 1/8 correct (12.5% accuracy)

**Why it failed:**
- **Hallucination Risk:** Gemini invented product names when OCR was unclear
  - Example: OCR said "Brigge CORN FLAKES", Gemini guessed "Kellogg's Corn Flakes"
- No visual context to ground the AI's predictions
- Unreliable for production use

**Lesson Learned:** LLMs need visual grounding to prevent hallucination.

---

### 4. CLIP-Only Recognition (BASELINE - 48% Accuracy)
**Files:**
- `services/recognition_service.py`

**Approach:** Use CLIP vision model to classify products into 68 predefined categories.

**Result:** Approximately 48% accuracy on products with clear visual features.

**Why it's limited:**
- Returns generic categories ("Sneakers") instead of specific products ("Nike Air Jordan 1")
- No text extraction capability
- Cannot identify specific brands or product variants
- Good for categorization, poor for precise product identification

**Lesson Learned:** Visual classification alone is insufficient for specific product recognition.

---

### 5. Enhanced Recognition (INTERMEDIATE - ~60% Accuracy)
**Files:**
- `services/enhanced_recognition_service.py`

**Approach:** Combine CLIP categories with OCR (Tesseract/EasyOCR) text extraction.

**Result:** ~60% accuracy (estimated)

**Why it's limited:**
- OCR extracts ALL text (marketing slogans, nutrition facts, etc.)
- No intelligent fusion of visual + text information
- Simple text extraction without understanding context
- Better than CLIP-only but still not production-ready

**Lesson Learned:** Simple concatenation of CLIP + OCR doesn't solve the core problem.

---

## Final Solution: Multimodal Pipeline (SUCCESS - 80% Accuracy)

After these experiments, we implemented a multimodal approach combining:

1. **Image Quality Check** - Reject poor images early
2. **CLIP Visual Analysis** - Understand product type visually
3. **OCR Text Extraction** - Get text from image (PaddleOCR)
4. **Gemini Multimodal** - Combine CLIP + OCR to extract accurate product name

**Files (Production):**
- `app/services/complete_recognition_service.py`
- `app/services/multimodal_product_extractor.py`
- `app/utils/image_quality.py`
- `app/utils/clip_describer.py`
- `tests/test_complete_pipeline.py`

**Result:** 8/10 correct (80% accuracy)

**Why it succeeded:**
- CLIP provides visual ground truth (prevents hallucination)
- OCR provides textual details (brand, product name)
- **Gemini fuses both modalities intelligently** (key difference from approach #5)
- Quality check filters out unusable images
- AI understands context and filters out marketing text

---

## Accuracy Progression

| Approach | Accuracy | Status |
|----------|----------|--------|
| Raw OCR (PaddleOCR) | 48.0% | Baseline |
| CLIP-Only Recognition | 48.0% | ⚠️ Limited (categories only) |
| GLiNER NER | 0.0% | ❌ Experiment Failed |
| Text Cleaning | 25.0% | ❌ Experiment Failed |
| Gemini OCR-only | 12.5% | ❌ Experiment Failed (Hallucination) |
| Enhanced Recognition (CLIP + OCR) | ~60.0% | ⚠️ Intermediate (no AI fusion) |
| **Complete Multimodal Pipeline** | **80.0%** | ✅ **Production Solution** |

---

## Key Insights

1. **Multimodal > Unimodal:** Combining visual (CLIP) + text (OCR) is superior to either alone
2. **Visual Grounding Prevents Hallucination:** CLIP prevents Gemini from guessing wrong products
3. **Simple Rules Don't Scale:** Hardcoded patterns can't handle unknown products
4. **NER Needs Structure:** Zero-shot NER fails on unstructured OCR output
5. **Quality Matters:** Pre-filtering poor images improves overall accuracy

---

## Documentation Value

These experimental files demonstrate:
- **Iterative development process**
- **Evidence-based decision making**
- **Transparent failure analysis**
- **Scientific approach to problem-solving**

---

