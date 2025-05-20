# LLM_tool.py
import os
import time
import gc
import sys
import torch
from threading import Thread
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    AutoConfig,
)

try:
    import bitsandbytes as bnb
    BNB_OK = True
except ImportError:
    BNB_OK = False

# 이미 내부 GPTQ/AWQ 양자화가 되어 있는 모델 모음
HEAVY_MODELS = {
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
    # 필요시 추가
}

# 전역 변수(필요시)
tok = None
model = None
DEVICE_MAP = None
USE_GPU = torch.cuda.is_available()


def _setup_cuda_env():
    """
    여러 GPU가 있을 경우 원하는 GPU를 고정하고 싶다면
    아래와 같이 환경변수를 설정할 수 있습니다.
    """
    if USE_GPU:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 예: 0번 카드 사용
        return "auto"
    else:
        return {"": "cpu"}  # 전부 CPU 할당


DEVICE_MAP = _setup_cuda_env()

def is_awq(model_id: str) -> bool:
    # 파일 이름·HuggingFace tag 기준 간단 판정
    return "AWQ" in model_id.upper()
def is_already_quantized(model_id: str) -> bool:
    """
    모델 config를 보고 GPTQ/AWQ 등
    내부적으로 이미 양자화가 되어 있는지 확인
    """
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return getattr(cfg, "quantization_config", None) is not None
    except Exception:
        # safetensors / GGUF 등 config가 없을 수도 있음
        return False


def max_gen_for(model, prompt_ids, hard_cap=None):
    """
    모델이 지원하는 최대 position embeddings 범위를 넘지 않도록
    안전한 토큰 수를 계산해주는 함수
    """
    ctx = getattr(model.config, "max_position_embeddings", 2048)
    room = ctx - len(prompt_ids)
    return min(room, hard_cap or room)

def load_model(mid: str, ltype: str):
    print(f"\n[Load] {mid} | {ltype}")
    kwargs = dict(device_map=DEVICE_MAP, trust_remote_code=True)

    # ── GPU일 때만 bnb 양자화 사용 ────────────────────────
    if USE_GPU and BNB_OK:
        if ltype == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(
                    torch.float16 if mid in HEAVY_MODELS else torch.bfloat16
                ),
            )
        elif ltype == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_compute_dtype=torch.bfloat16
            )
        elif ltype in ["fp16", "bf16"]:
            kwargs["torch_dtype"] = torch.float16 if ltype == "fp16" else torch.bfloat16
        elif ltype == "fp8":
            kwargs["torch_dtype"] = torch.float8_e4m3fn
        elif ltype == "awq":
            pass
        else:
            raise ValueError(f"Unknown load type: {ltype}")

    if is_awq(mid):
        # bnb 양자화와 충돌 가능성 → 제거
        kwargs.pop("quantization_config", None)
        # torch_dtype='auto' 로 두면 로더가 알아서 FP16/BF16 로 맞춤
        kwargs["torch_dtype"] = "auto"

    # ── CPU fallback ──────────────────────────────────────
    else:
        # bnb 옵션 제거
        kwargs.pop("quantization_config", None)
        # fp16/bf16 → CPU 지원이 미묘하니 안전하게 float32
        kwargs["torch_dtype"] = torch.float32
        # device_map 이 cpu이면 AutoHF가 알아서 .to("cpu") 처리

    if is_already_quantized(mid):
        kwargs.pop("quantization_config", None)  # GPTQ·AWQ 충돌 방지
        if ltype in ["4bit", "8bit"]:
            print("  ↳  내부 양자화 모델 : fp16 로 재시도");
            kwargs["torch_dtype"] = torch.float16

    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    tok.pad_token_id = tok.pad_token_id or tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(mid, **kwargs).eval()
    return tok, model

@torch.inference_mode()
def generate(tok, model, prompt: str, max_new_tokens: int = 128):
    """
    스트리밍 없이 한 번에 출력을 반환하는 버전.
    """
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    # 프롬프트 이후만 추출
    return txt.split(prompt, 1)[-1].strip()


@torch.inference_mode()
def generate_stream(tok, model, prompt: str, max_new_tokens: int = 128):
    """
    스트리머를 활용하여 토큰 단위로 실시간 확인 가능한 버전.
    여기서는 최종 결과만 모아서 return.
    """
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True)

    thread = Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        ),
        daemon=True
    )
    thread.start()

    out_text = ""
    with tqdm(total=max_new_tokens, desc="Generating", file=sys.stderr, disable=False) as pbar:
        for piece in streamer:
            out_text += piece
            pbar.update(1)

    return out_text.split(prompt, 1)[-1].lstrip()


def free_gpu(*objs):
    """
    GPU 메모리 정리를 위한 함수.
    """
    for o in objs:
        try:
            del o
        except:
            pass
    if USE_GPU:  # GPU가 있을 때만
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def init_model(model_id: str = "Qwen/Qwen3-14B-FP8",
               load_type: str = "fp8"):
    """
    전역 변수 tok, model에 로드.
    여러 번 호출하면 매번 모델을 갈아끼우는 형태.
    """
    global tok, model
    # 기존 모델 있으면 먼저 메모리 정리
    if tok or model:
        free_gpu(model, tok)
    # 새로 로드
    tok, model = load_model(model_id, load_type)


def get_LLM_response(prompt: str,
                     model_id: str = "Qwen/Qwen3-14B-FP8",
                     load_type: str = "fp8",
                     max_new_tokens: int = 7950,
                     stream: bool = True) -> str:
    """
    - prompt: 유저가 넣는 프롬프트
    - model_id/load_type: 원하는 모델과 로드 방식
    - max_new_tokens: 생성 토큰수 제한
    - stream: 스트림 형태로 받을지 여부
    """

    # 만약 모델이 전혀 로드 안 되어 있거나,
    # model_id/load_type이 현재 전역과 다른 경우 재초기화
    global tok, model

    # 간단 검증: 아직 모델이 없으면 init
    if not model or not tok:
        init_model(model_id, load_type)

    # 이미 로드된 모델이 원하는 모델이 아닐 수도 있음(추가 로직 가능)
    # 예) 여기서는 단순하게 항상 init_model() 하도록 처리해도 됨.

    # 실제 토큰 최대치 계산 (optional)
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
    safe_tokens = max_gen_for(model, prompt_ids)
    final_tokens = min(len(prompt), safe_tokens)
    if not stream:
        ans = generate(tok, model, prompt, final_tokens)
    else:
        ans = generate_stream(tok, model, prompt, final_tokens)

    return ans

if __name__ == "__main__":
    prompt = "오늘날 금융시장에 대해 300자 이내로 이야기해줘."

    # 스트리밍 형태:
    ans_stream = get_LLM_response(prompt, stream=True)
    print(ans_stream)