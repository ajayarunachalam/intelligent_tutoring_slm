# Tutoring SLM Engine — DGX Spark Proof of Concept
## Built on MathDial Dataset + Gemma 4 (via Ollama)

### Architecture
```
MathDial Dataset
      ↓
  data_loader.py      — Download, parse, split MathDial
  preprocessor.py     — Format into Gemma chat template
  fine_tuner.py       — QLoRA fine-tuning via Unsloth/TRL
  inference.py        — Ollama-based local inference engine
  tutor_engine.py     — Core Socratic tutoring logic
  evaluator.py        — BLEU, solve-rate, pedagogy metrics
  app.py              — Interactive CLI tutor session
  pipeline.py         — Full end-to-end runner
```

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (download data → fine-tune → evaluate → demo)
python pipeline.py --mode full

# 3. Run interactive tutor (uses base Gemma 4 via Ollama)
python pipeline.py --mode demo

# 4. Fine-tune only
python pipeline.py --mode finetune --base-model gemma4:latest

# 5. Evaluate only
python pipeline.py --mode evaluate
```

### Models (already on DGX Spark)
- `gemma4:latest`  — fast prototyping & interactive demo
- `gemma4:31b`     — full fine-tuning & production quality
