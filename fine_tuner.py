"""
fine_tuner.py
─────────────
QLoRA fine-tuning of Gemma 4 on the MathDial tutoring dataset.
Optimised for the DGX Spark GB10 Grace Blackwell SoC (128 GB unified memory).

Strategy:
- Load Gemma 4 in 4-bit NF4 (bitsandbytes) via QLoRA
- Apply LoRA adapters to attention + FFN projection layers
- Train with TRL SFTTrainer on the preprocessed MathDial chat format
- Save merged model + standalone adapter for efficient inference

DGX Spark notes:
- GB10 has 128 GB unified CPU+GPU memory → no CPU offloading needed
- Use bf16 (Blackwell supports native bf16)
- FSDP not needed for single-node, but gradient checkpointing helps
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

console = Console()
logger = logging.getLogger(__name__)


# ── Config dataclasses ─────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 16                         # rank
    lora_alpha: int = 32                # scaling = alpha / r
    lora_dropout: float = 0.05
    bias: str = "none"
    # Target modules for Gemma 4 architecture
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Full training configuration."""
    # Model
    base_model_name: str        = "google/gemma-2-9b-it"   # HF name (or local path)
    ollama_model_name: str      = "gemma4:latest"           # Ollama name on DGX Spark
    use_4bit_quantisation: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"               # bf16 on Blackwell
    bnb_4bit_quant_type: str    = "nf4"

    # Output
    output_dir: str             = "./checkpoints/tutoring_slm"
    final_model_dir: str        = "./models/tutoring_slm_final"
    adapter_dir: str            = "./models/tutoring_slm_adapter"

    # Training hyperparameters (tuned for DGX Spark 128GB)
    num_train_epochs: int       = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int  = 4
    gradient_accumulation_steps: int = 4    # effective batch = 4 * 4 = 16
    learning_rate: float        = 2e-4
    lr_scheduler_type: str      = "cosine"
    warmup_ratio: float         = 0.05
    max_seq_length: int         = 2048
    weight_decay: float         = 0.01

    # Efficiency
    gradient_checkpointing: bool = True
    fp16: bool                  = False
    bf16: bool                  = True      # Blackwell native

    # Evaluation & saving
    eval_strategy: str          = "steps"
    eval_steps: int             = 100
    save_steps: int             = 200
    logging_steps: int          = 25
    load_best_model_at_end: bool = True
    metric_for_best_model: str  = "eval_loss"

    # Misc
    seed: int                   = 42
    report_to: str              = "none"    # set to "wandb" for tracking


# ── Fine-tuner ─────────────────────────────────────────────────────────────

class TutoringSLMFineTuner:
    """
    End-to-end QLoRA fine-tuner for the Tutoring SLM.

    Usage:
        ft = TutoringSLMFineTuner(training_cfg, lora_cfg)
        ft.train(train_dataset, eval_dataset)
        ft.save()
    """

    def __init__(
        self,
        training_config: Optional[TrainingConfig] = None,
        lora_config:     Optional[LoRAConfig]     = None,
    ):
        self.cfg  = training_config or TrainingConfig()
        self.lcfg = lora_config     or LoRAConfig()
        self.model     = None
        self.tokenizer = None
        self.trainer   = None

    # ── Public API ──────────────────────────────────────────────────────────

    def train(self, train_dataset: Dataset, eval_dataset: Dataset) -> None:
        """Run the full fine-tuning pipeline."""
        console.print("\n[bold cyan]═══ TutoringSLM Fine-tuning ═══[/bold cyan]")
        self._log_device_info()
        self._load_model_and_tokenizer()
        self._apply_lora()
        self._run_sft(train_dataset, eval_dataset)

    def save(self) -> None:
        """Save the final adapter and optionally the merged model."""
        if self.trainer is None:
            raise RuntimeError("Call train() before save()")

        adapter_path = Path(self.cfg.adapter_dir)
        adapter_path.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[cyan]Saving LoRA adapter → {adapter_path}[/cyan]")
        self.trainer.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))

        # Save training config alongside the adapter
        cfg_path = adapter_path / "training_config.json"
        with open(cfg_path, "w") as f:
            json.dump(self.cfg.__dict__, f, indent=2)

        console.print("[green]✓ Adapter saved.[/green]")
        console.print(
            f"\n[bold]To use in Ollama, merge and convert:[/bold]\n"
            f"  python fine_tuner.py --merge-adapter {adapter_path}"
        )

    def merge_and_save(self) -> None:
        """Merge LoRA weights into base model and save full model."""
        if self.model is None:
            self._load_model_and_tokenizer()

        console.print("\n[cyan]Merging LoRA adapters into base model...[/cyan]")
        from peft import PeftModel
        merged = self.model.merge_and_unload()

        out_path = Path(self.cfg.final_model_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out_path), safe_serialization=True)
        self.tokenizer.save_pretrained(str(out_path))

        console.print(f"[green]✓ Merged model saved → {out_path}[/green]")
        console.print(
            "\n[bold]To register in Ollama:[/bold]\n"
            f"  ollama create tutoring-slm -f Modelfile\n"
            "  (See Modelfile generated in the model directory)"
        )
        self._generate_modelfile(out_path)

    # ── Private methods ──────────────────────────────────────────────────────

    def _log_device_info(self):
        """Print GPU/CPU memory info for DGX Spark."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(f"[green]GPU: {gpu_name}  |  Total VRAM: {total_mem:.1f} GB[/green]")
        else:
            console.print("[yellow]No CUDA GPU found — training on CPU (slow)[/yellow]")

        console.print(f"[dim]Base model:  {self.cfg.base_model_name}[/dim]")
        console.print(f"[dim]Output dir:  {self.cfg.output_dir}[/dim]")
        console.print(f"[dim]Quantisation: {'4-bit NF4' if self.cfg.use_4bit_quantisation else 'full precision'}[/dim]")

    def _load_model_and_tokenizer(self):
        """Load base model with optional 4-bit quantisation."""
        console.print(f"\n[cyan]Loading model: {self.cfg.base_model_name}[/cyan]")

        # BitsAndBytes 4-bit config
        bnb_cfg = None
        if self.cfg.use_4bit_quantisation:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.cfg.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.base_model_name,
            trust_remote_code=True,
            padding_side="right",
        )

        # Gemma uses <pad> = <eos> by convention
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        console.print(f"[green]✓ Model loaded.[/green]")

    def _apply_lora(self):
        """Wrap the model with LoRA adapters."""
        if self.cfg.use_4bit_quantisation:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.cfg.gradient_checkpointing,
            )

        lora_cfg = LoraConfig(
            r=self.lcfg.r,
            lora_alpha=self.lcfg.lora_alpha,
            target_modules=self.lcfg.target_modules,
            lora_dropout=self.lcfg.lora_dropout,
            bias=self.lcfg.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

    def _run_sft(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Instantiate SFTTrainer and run training."""
        training_args = SFTConfig(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.num_train_epochs,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            lr_scheduler_type=self.cfg.lr_scheduler_type,
            warmup_ratio=self.cfg.warmup_ratio,
            weight_decay=self.cfg.weight_decay,
            fp16=self.cfg.fp16,
            bf16=self.cfg.bf16,
            gradient_checkpointing=self.cfg.gradient_checkpointing,
            eval_strategy=self.cfg.eval_strategy,
            eval_steps=self.cfg.eval_steps,
            save_steps=self.cfg.save_steps,
            logging_steps=self.cfg.logging_steps,
            load_best_model_at_end=self.cfg.load_best_model_at_end,
            metric_for_best_model=self.cfg.metric_for_best_model,
            seed=self.cfg.seed,
            report_to=self.cfg.report_to,
            max_seq_length=self.cfg.max_seq_length,
            dataset_text_field="text",
            packing=False,   # False for tutoring: preserve dialogue structure
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        console.print("\n[bold cyan]Starting training...[/bold cyan]")
        self.trainer.train()
        console.print("[green]✓ Training complete.[/green]")

    def _generate_modelfile(self, model_path: Path):
        """Generate an Ollama Modelfile for the fine-tuned model."""
        from preprocessor import TUTOR_SYSTEM_PROMPT
        modelfile_content = f"""FROM {model_path}

SYSTEM \"\"\"{TUTOR_SYSTEM_PROMPT.split(chr(10))[0]}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 256
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"
"""
        modelfile_path = model_path / "Modelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        console.print(f"[green]✓ Modelfile written → {modelfile_path}[/green]")


# ── Convenience function ───────────────────────────────────────────────────

def finetune_from_disk(
    data_dir: str = "./data/processed",
    model_name: str = "google/gemma-2-9b-it",
    quick_run: bool = False,   # True = 1 epoch + tiny dataset for smoke-test
) -> TutoringSLMFineTuner:
    """
    Load preprocessed data from disk and run fine-tuning.

    Args:
        data_dir:   path to saved HuggingFace datasets
        model_name: HuggingFace model ID or local path
        quick_run:  smoke-test mode (fast, 1 epoch)
    """
    train_ds = load_from_disk(f"{data_dir}/train")
    eval_ds  = load_from_disk(f"{data_dir}/test")

    if quick_run:
        train_ds = train_ds.select(range(min(200, len(train_ds))))
        eval_ds  = eval_ds.select(range(min(50, len(eval_ds))))
        console.print(f"[yellow]Quick run: using {len(train_ds)} train / {len(eval_ds)} eval samples[/yellow]")

    cfg = TrainingConfig(
        base_model_name=model_name,
        num_train_epochs=1 if quick_run else 3,
        per_device_train_batch_size=2 if quick_run else 4,
    )

    ft = TutoringSLMFineTuner(training_config=cfg)
    ft.train(train_ds, eval_ds)
    ft.save()
    return ft


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Tutoring SLM")
    parser.add_argument("--model",       default="google/gemma-2-9b-it", help="HF model name or path")
    parser.add_argument("--data-dir",    default="./data/processed")
    parser.add_argument("--quick",       action="store_true", help="Smoke-test run (1 epoch, 200 samples)")
    parser.add_argument("--merge-adapter", type=str, help="Path to adapter to merge into base model")
    args = parser.parse_args()

    if args.merge_adapter:
        ft = TutoringSLMFineTuner()
        ft._load_model_and_tokenizer()
        from peft import PeftModel
        ft.model = PeftModel.from_pretrained(ft.model, args.merge_adapter)
        ft.merge_and_save()
    else:
        finetune_from_disk(
            data_dir=args.data_dir,
            model_name=args.model,
            quick_run=args.quick,
        )
