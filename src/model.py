"""Model/tokenizer loading with LoRA (and optional 4-bit QLoRA) setup."""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str,
    model_revision: str = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules=None,
    use_4bit: bool = False,
    device: str = "auto",
):
    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_kwargs = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    use_cuda = torch.cuda.is_available() and device != "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        **quant_kwargs,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if not use_cuda:
        model = model.to("cpu")

    return model, tokenizer
