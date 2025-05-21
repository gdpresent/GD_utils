# LLM_tool.py

__default_LLM_model__      = "Qwen/Qwen3-14B-FP8"
__default_LLM_load_type__  = "fp8"

__default_VLM_model__      = "Qwen/Qwen2.5-VL-7B-Instruct"
__default_VLM_load_type__  = "4bit"

import os
import time
import gc
import sys
import torch
import traceback
from threading import Thread
from tqdm.auto import tqdm
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
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

# 전역 변수
tok = None
model = None
vlm_processor = None
vlm_model = None
VLM_MODEL_ID = None
VLM_LOAD_TYPE = None

DEVICE_MAP = None
USE_GPU = torch.cuda.is_available()


def _setup_cuda_env(gpu_index=None):
    if USE_GPU:
        if gpu_index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # 이미 환경 변수가 설정되어 있다면 굳이 setdefault로 고정하지 않도록
        return "auto"
    else:
        return {"": "cpu"}
DEVICE_MAP = _setup_cuda_env()

def is_awq(model_id: str) -> bool:
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

def free_gpu(*objs):
    """GPU 메모리 정리용"""
    for o in objs:
        try:
            del o
        except:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
def vram():
    """GPU 메모리 상황을 문자열로 반환"""
    free, tot = torch.cuda.mem_get_info()
    return f"{(tot-free)/2**20:,.0f} MB / {tot/2**20:,.0f} MB"

def load_LLM_model(mid: str, ltype: str):
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
def load_VLM_model(model_id: str, load_type: str = "bf16"):
    """
    Qwen2.5-VL 계열 모델 로드 함수 예시.
    AutoModelForCausalLM 대신 Qwen2_5_VLForConditionalGeneration을 사용해야 함.
    """
    print(f"\n[Load] {model_id} | {load_type}")
    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = None

        if load_type == "4bit" and BNB_OK:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device_map,
                quantization_config=bnb_config,
                trust_remote_code=True
            )
        elif load_type == "8bit" and BNB_OK:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_compute_dtype=torch.bfloat16
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device_map,
                quantization_config=bnb_config,
                trust_remote_code=True
            )
        elif load_type in ["fp16", "bf16"]:
            torch_dtype = torch.float16 if load_type == "fp16" else torch.bfloat16
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
        else:
            print(f"  [warn] Unrecognized load_type={load_type}, default to bf16.")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
    else:
        # CPU 모드
        print("  [info] CPU mode")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

    # processor: 이미지+비디오+텍스트를 같이 전처리
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True    # Warning 메시지 없애려면 명시적으로 False (혹은 True)
    )
    model.eval()
    return processor, model
def load_local_image(image_path: str):
    """
    로컬 이미지를 PIL Image로 로드해주는 간단한 헬퍼 함수
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    # 모델에 맞춰서 RGB로 변환
    return Image.open(image_path).convert("RGB")

def init_LLM_model(model_id: str = None, load_type: str = None):
    """
    전역 변수 tok, model에 로드.
    여러 번 호출하면 매번 모델을 갈아끼우는 형태.
    """
    global tok, model

    if model_id is None:
        model_id = __default_LLM_model__
    if load_type is None:
        load_type = __default_LLM_load_type__

    # 기존 모델 있으면 먼저 메모리 정리
    if tok or model:
        free_gpu(model, tok)
    # 새로 로드
    tok, model = load_LLM_model(model_id, load_type)
def init_VLM_model(model_id: str = None, load_type: str = None):
    """
    전역 변수 vlm_processor, vlm_model에 로드.
    여러 번 호출하면 매번 모델을 갈아끼우는 형태.
    """
    global vlm_processor, vlm_model

    if model_id is None:
        model_id = __default_VLM_model__
    if load_type is None:
        load_type = __default_VLM_load_type__

    # 기존 모델이 있으면 메모리 정리
    if vlm_processor or vlm_model:
        free_gpu(vlm_model, vlm_processor)

    # 새로 로드
    vlm_processor, vlm_model = load_VLM_model(model_id, load_type)

@torch.inference_mode()
def generate_byLLM(tok, model, prompt: str, max_new_tokens: int = 128):
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
def generate_byLLM_stream(tok, model, prompt: str, max_new_tokens: int = 128):
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

@torch.inference_mode()
def generate_caption_byVLM(processor, model, image_path: str, prompt: str, max_new_tokens: int = 128):
    """
    로컬 이미지 + 텍스트를 받아서 Qwen2.5-VL 모델로부터 답변을 생성.
    """
    # 1) 메시지 포맷: Qwen-VL은 'messages' 구조를 사용
    #   - 실제 이미지 데이터는 processor(...)에 따로 넣는다.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # 2) 최종 토큰화를 위한 텍스트 구성
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 3) 로컬 이미지 로드
    img = load_local_image(image_path)

    # 4) Processor를 통한 토큰화
    inputs = processor(
        text=[text],
        images=[img],
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # 5) generate
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 6) 프롬프트 길이만큼 잘라내고 디코딩
    generated_ids_trimmed = []
    for in_ids, out_ids in zip(inputs["input_ids"], outputs):
        new_ids = out_ids[len(in_ids):]
        generated_ids_trimmed.append(new_ids)

    captions = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return captions[0] if len(captions) > 0 else "(empty)"

@torch.inference_mode()
def generate_caption_byVLM_stream(processor, model, image_path: str, prompt: str, max_new_tokens: int = 128):
    """
    스트리밍 방식(토큰-by-토큰)으로 문장 생성 예시.
    """
    # 1) 메시지
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img = load_local_image(image_path)

    # 2) 입력 만들기
    inputs = processor(
        text=[text],
        images=[img],
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # 3) 스트리밍(직접 루프) 방식
    generated = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    pbar = tqdm(total=max_new_tokens, desc="토큰 생성 중")
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        pbar.update(1)
    pbar.close()

    # 4) 새로 생성된 토큰만 디코딩
    new_tokens = generated[:, inputs["input_ids"].size(1):]
    caption = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return caption



def get_LLM_response(prompt: str, model_id: str = None, load_type: str = None, max_new_tokens: int = 7950, stream: bool = False) -> str:
    """
    - prompt: 유저가 넣는 프롬프트
    - model_id/load_type: 원하는 모델과 로드 방식
    - max_new_tokens: 생성 토큰수 제한
    - stream: 스트림 형태로 받을지 여부
    """

    # 만약 모델이 전혀 로드 안 되어 있거나,
    # model_id/load_type이 현재 전역과 다른 경우 재초기화
    global tok, model
    if model_id is None:
        model_id = __default_LLM_model__
    if load_type is None:
        load_type = __default_LLM_load_type__

    # 간단 검증: 아직 모델이 없으면 init
    if not model or not tok:
        init_LLM_model(model_id, load_type)

    # 실제 토큰 최대치 계산 (optional)
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
    safe_tokens = max_gen_for(model, prompt_ids)
    final_tokens = max(max_new_tokens, safe_tokens)
    if not stream:
        stt_time = time.time()
        ans = generate_byLLM(tok, model, prompt, final_tokens)
        print(f"LLM time: {time.time() - stt_time:.3f}s")
    else:
        ans = generate_byLLM_stream(tok, model, prompt, final_tokens)
    print(ans)
    return ans

def get_VLM_response(image_path: str, prompt: str, model_id: str = None, load_type: str = None, max_new_tokens: int = 7950, stream: bool = False) -> str:
    """
    - image_path: 로컬 이미지 경로
    - prompt: 유저 텍스트
    - model_id / load_type: 원하는 모델 / 로드 방식
    - max_new_tokens: 생성 제한
    - stream: 스트리밍 여부
    """
    global vlm_processor, vlm_model
    if model_id is None:
        model_id = __default_VLM_model__
    if load_type is None:
        load_type = __default_VLM_load_type__

    # 아직 모델이 없거나, 다른 모델이면 재초기화
    if (vlm_model is None) or (vlm_processor is None):
        init_VLM_model(model_id, load_type)

    if not stream:
        stt_time = time.time()
        ans = generate_caption_byVLM(vlm_processor, vlm_model, image_path, prompt, max_new_tokens)
        print(f"VLM time: {time.time() - stt_time:.3f}s")
        return ans
    else:
        ans = generate_caption_byVLM_stream(vlm_processor, vlm_model, image_path, prompt, max_new_tokens)
        return ans

if __name__ == "__main__":
    # IMAGE_PATH = "C:/GD_GIT/GD_Crawling/Crawling_FnGuide/images/chart/6a7b9f07_c_4_32.png"
    IMAGE_PATH = "C:/GD_GIT/GD_Crawling/Crawling_FnGuide/images/figure/3d5faac9_f_6_55.png"
    PROMPT = "이 이미지에 대해 자세히 설명해줘."
    ans_VLM = get_VLM_response(IMAGE_PATH, PROMPT)
    print(ans_VLM)

    prompt = "오늘날 금융시장에 대해 300자 이내로 이야기해줘."
    ans_LLM = get_LLM_response(prompt)
    print(ans_stream)

