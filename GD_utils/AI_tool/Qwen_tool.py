# Qwen_tool.py

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
import re
from threading import Thread
from tqdm.auto import tqdm
from PIL import Image
from GD_utils.AI_tool.NLP_tool import split_sentences

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
        torch.cuda.synchronize()
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
    global tok, model, LLM_MODEL_ID, LLM_LOAD_TYPE

    model_id  = model_id  or __default_LLM_model__
    load_type = load_type or __default_LLM_load_type__

    # 이미 떠 있는데 ID 또는 로드 방식이 바뀌면 언로드
    if model is not None and (model_id != LLM_MODEL_ID or load_type != LLM_LOAD_TYPE):
        _unload_LLM()

    if model is None:  # 실제 로드
        tok, model = load_LLM_model(model_id, load_type)
        LLM_MODEL_ID, LLM_LOAD_TYPE = model_id, load_type

def init_VLM_model(model_id: str = None, load_type: str = None):
    global vlm_processor, vlm_model, VLM_MODEL_ID, VLM_LOAD_TYPE

    model_id  = model_id  or __default_VLM_model__
    load_type = load_type or __default_VLM_load_type__

    if vlm_model is not None and (model_id != VLM_MODEL_ID or load_type != VLM_LOAD_TYPE):
        _unload_VLM()

    if vlm_model is None:
        vlm_processor, vlm_model = load_VLM_model(model_id, load_type)
        VLM_MODEL_ID, VLM_LOAD_TYPE = model_id, load_type

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
        early_stopping=True,
        repetition_penalty=1.2,
        top_p=0.9,
        do_sample=True,
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
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
            early_stopping=True,
            repetition_penalty=1.2,
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


def clean_qwen_output(raw: str) -> str:
    """
    1) <think> 블록 제거
    2) 마지막 'assistant' 헤더 이후만 반환
    """
    _THINK_BLOCK = re.compile(r".*?</think>", flags=re.S)

    # ① 내부 사고 블록 제거
    txt = _THINK_BLOCK.sub("", raw)

    # ② 가장 마지막 'assistant' 한 줄 찾기
    last = None
    for m in re.finditer(r"^assistant\s*$", txt, flags=re.M):
        last = m.end()

    # ③ 있으면 그 뒤부터, 없으면 role 헤더 한 줄씩 삭제
    if last is not None:
        txt = txt[last:]
    else:
        txt = "\n".join(
            line for line in txt.splitlines()
            if not re.match(r"^\s*(system|user|assistant)\s*$", line)
        )
    parts = re.split(r"\n\s*\n", txt, maxsplit=1)  # 두 줄 연속 공백 기준 split
    output = parts[-1] if len(parts) > 1 else txt  # 없으면 원본

    return output.strip()
def get_LLM_response(prompt: str, model_id: str = None, load_type: str = None, max_new_tokens: int = 7950, stream: bool = False) -> str:
    """
    - prompt: 유저가 넣는 프롬프트
    - model_id/load_type: 원하는 모델과 로드 방식
    - max_new_tokens: 생성 토큰수 제한
    - stream: 스트림 형태로 받을지 여부
    """

    # 만약 모델이 전혀 로드 안 되어 있거나,
    global tok, model
    if model_id is None:
        model_id = __default_LLM_model__
    if load_type is None:
        load_type = __default_LLM_load_type__

    if not model or not tok:
        init_LLM_model(model_id, load_type)

    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
    safe_tokens = max_gen_for(model, prompt_ids)
    final_tokens = min(safe_tokens, len(prompt))
    if not stream:
        stt_time = time.time()
        ans = generate_byLLM(tok, model, prompt, final_tokens)
        print(f"LLM time: {time.time() - stt_time:.3f}s")
    else:
        ans = generate_byLLM_stream(tok, model, prompt, final_tokens)
    if 'qwen' in model_id.lower():
        ans = clean_qwen_output(ans)
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

def _unload_LLM():
    """LLM 메모리를 완전히 반환."""
    global tok, model, LLM_MODEL_ID, LLM_LOAD_TYPE
    if model is not None:
        model.to("cpu")
    print("[Unload] LLM")
    free_gpu(model, tok)
    tok = model = None
    LLM_MODEL_ID = LLM_LOAD_TYPE = None
def _unload_VLM():
    """VLM 메모리를 완전히 반환."""
    global vlm_processor, vlm_model, VLM_MODEL_ID, VLM_LOAD_TYPE
    if vlm_model is not None:
        vlm_model.to("cpu")
    print("[Unload] VLM")
    free_gpu(vlm_model, vlm_processor)
    vlm_processor = vlm_model = None
    VLM_MODEL_ID = VLM_LOAD_TYPE = None

def summarize_long_text(
        text: str,
        *,
        language: str = "auto",
        max_chunk_sents: int = 5,  # chunk 당 문장 최대 개수
        overlap_sents: int = 2,  # chunk 간 문장 중복 개수
        max_new_tokens: int = 1024,  # 1회 요약 시 생성 토큰 최대치
        first_pass_prompt_template: str = (
                "아래 글을 간결하고 핵심만 요약해줘.\n\n"
                "글:\n{chunk_text}\n"
                "요약:"
        ),
        final_pass_prompt_template: str = (
                "다음은 여러 부분으로 나눠 요약된 결과입니다. 이들을 종합하여 "
                "최종적으로 더욱 간략하고 핵심 위주로 요약해 주세요.\n\n"
                "{all_summaries}\n\n"
                "통합 요약:"
        ),
        model_id: str = None,
        load_type: str = None,
) -> str:
    """
    긴 텍스트를 문장 단위로 분리한 뒤(overlap 포함) 여러 chunk로 나누어 LLM에 순차적으로 요약시키고,
    마지막에 부분 요약들을 다시 하나로 모아서 최종 요약을 도출한다.

    Parameters
    ----------
    text : str
        전체 원문 텍스트(아주 길 수도 있음).
    language : str
        split_sentences에 넘길 언어 옵션("auto"/"ko"/"en" 등).
    max_chunk_sents : int
        1차 요약 시, 한 chunk에 포함될 문장 최대 개수.
    overlap_sents : int
        chunk 간 중첩되는 문장 개수(맥락 유지용).
        ex) max_chunk_sents=5, overlap_sents=2 → [0..4], [3..7], [5..9], ...
    max_new_tokens : int
        요약 생성 시 최대 new tokens.
    first_pass_prompt_template : str
        1차 요약 프롬프트 템플릿. {chunk_text} 위치에 chunk가 들어감.
    final_pass_prompt_template : str
        최종 통합 요약 프롬프트 템플릿. {all_summaries} 위치에 1차 요약 리스트가 합쳐진 텍스트가 들어감.
    model_id : str
        사용할 LLM ID (없으면 Qwen_tool.py 상의 __default_LLM_model__).
    load_type : str
        로딩 방식 (없으면 __default_LLM_load_type__).

    Returns
    -------
    final_summary : str
        1차 요약들을 최종 통합한 최종 요약문.
    """

    # 1) 문장 분리
    sents = split_sentences(text, language=language)
    if not sents:
        return ""

    # 2) 문장 chunk 만들기 (overlap 적용)
    #    예: sents = 10개, max_chunk_sents=5, overlap_sents=2 → chunks = [[0..4], [3..7], [5..9]]
    chunks = []
    start_idx = 0
    while True:
        end_idx = start_idx + max_chunk_sents
        # 문장들을 모아서 하나의 chunk_text로 만든다.
        chunk_sents = sents[start_idx:end_idx]
        if not chunk_sents:
            break
        chunk_text = "\n".join(chunk_sents)
        chunks.append(chunk_text)

        if end_idx >= len(sents):
            break
        # 오버랩 적용 → 다음 chunk의 시작 인덱스는 (end_idx - overlap_sents)
        start_idx = end_idx - overlap_sents
        if start_idx < 0:
            start_idx = 0

    # 3) 각 chunk 별 부분 요약
    partial_summaries = []
    for i, ctext in enumerate(chunks, 1):
        # 프롬프트 구성
        prompt = first_pass_prompt_template.format(chunk_text=ctext.strip())
        # LLM 호출 (LLM_tool.py의 get_LLM_response)
        summary_i = get_LLM_response(
            prompt,
            model_id=model_id,
            load_type=load_type,
            max_new_tokens=max_new_tokens,
            stream=False  # 스트리밍 필요 없으면 False
        )
        partial_summaries.append(summary_i.strip())

    # 4) 부분 요약들을 다시 하나로 모아 최종 요약
    #    - 부분 요약의 길이가 여전히 길다면, 또다시 chunking하여 2차/3차 요약할 수도 있음.
    #    - 여기서는 단순히 한 번에 다 붙여 최종 요약 프롬프트를 만든다.
    merged_partial = "\n\n".join(
        f"[부분요약 {idx}]\n{summ}"
        for idx, summ in enumerate(partial_summaries, 1)
    )
    final_prompt = final_pass_prompt_template.format(all_summaries=merged_partial)
    final_summary = get_LLM_response(
        final_prompt,
        model_id=model_id,
        load_type=load_type,
        max_new_tokens=max_new_tokens,
        stream=False
    )
    return final_summary.strip()


if __name__ == "__main__":
    ko_example_to_be_summarized = "지난 주말 스위스 제네바에서 첫 공식 무역 협상을 마치고 미·중 양국은 90일간 상호 부과한 고율 관세를 대폭 인하하기로 합의했다고 밝힌 가운데, 미·중간 무역 갈등 완화 기대감이 지속된 점이 투자심리를 개선시키는 모습을 보인 가운데, 외국인, 기관 동반 순매도 등에 코스닥지수는 1% 넘게 하락하며 지수 상승을 이끌었으며 외국인은 8거래일 연속 순매수, 기관은 5거래일만에 순매수로 전환했다. 한편, 무디스는 신용등급 강등의 이유로 미국 정부의 만성적인 부채 증가와 이자 부담을 지적했으며, 월가 애널리스트들은 무역 협상 등을 주목하며 관망 심리가 악화할 것이라는 우려가 완화되고 있다. 미국 경제의 불확실성이 완화될 것이라는 기대감이 지속되어 투자심리가 개선될 것으로 보인다.  국내증시 코피스 지수는 0.21% 상승한 2626.87에 마감했다.   미국증시 5월 16일 뉴욕증시는 옵션 만기일을 맞이한 가운데, 단기 상승 모멘트가 부재한 가운데, 외국인과 기관이 동반 매수했다. 미·미·중 무역 갈등 완화에 대한 기대감 및 미·무역 갈등 완화 기대감 등을 우려하고 있다. 미국과 중국이 90일간 서로 부과한 관세를 낮추기로 합의했다고 밝혔으며, 미국과 중국의 관세 인하에 대한 우려가 완화될 것으로 예상된다. 미국, 미국, 중국, 중국 등 세계 경제가 악화할 수 있다는 우려에 대한 우려에 대해 우려로 상승할 것으로 예상되며, 미국 증시가 상승세로 전환될 것으로 전망된다. 한편 미·중국과의 무역갈등 완화를 기대하며 투자심리에 영향을 미칠 것으로 보고 있다. 미-중 무역갈등이 완화될 것이라고 전망했다. 미국-미중 무역 갈등이 완화될 것이란 기대감에 투자 심리가 완화되고 있는 가운데, 미국과 중국 양국의 무역갈등에 대한 우려로 인해 투자심리의 불확실성이 악화될 것이라는 우려로 투자심리는 지속되고 있어 투자심리로 전환될 수 있다는 우려가 커지고 있다. 한편 미국, 미중 무역분쟁 완화 기대감에 따라 투자심리도 개선될 것이라고 밝혔다. 미국 무역분쟁에 대한 불확실성이 커지고 있는 상황에서 투자 심리를 완화시킬 수 있을 것으로 보고 투자 심리에 대한 우려가 해소될 수 있을 것이라고 말했다. 한편 중국, 미국과의 무역 갈등이 해소될 것이라는 기대감에도 불구하고 미중 양국이 무역분쟁을 완화할 것이라는 기대감에 대한 시장의 불확실성이 높아지고 있다. 갭이 지속되고 있는 상황이다. 셧다운이 완화될 수 있는 상황이 지속되고 있다고 밝혔다. 랠리를 회복할 수 있을 것이라는 기대감으로 투자심리 회복이 지속될 수 있다는 것이 투자심리 개선으로 이어질 수 있을 것이다. 틸리스크에 대한 불안감이 해소될 것으로 보여진다. 챕터 랠리가 지속되는 가운데, 셧 다운을 완화시킬 것이라는 기대감이 커지고 있어 투자심리 완화를 위해 미국과 중국과의 무역전쟁이 완화되는지 랠리 랠리에서 벗어나는  에 대한 로 의 와 를 에서 , 라는 가 ()  및 적 을 라고 해석했다. 미국 이  반등하고 있다로 전환될지 도 고  상승 과  등  투자심리  증가는 증시 로 해석된다. 미국로 인해 미중  미·를 통해 미국로 인한 주 나 중의 불확실성 확대 제 완화  완화를 반영  완화 대가 완화될 것이다.가 될 수 있는 지를 완화로 작용할 수 있다.라는 미·대로 이어질 수 있다는 관세 세가 지속되고 있다가 지속될 수 있는 투자심리에 대해 투자로 작용해 거래소로 투자 시장로가 기대된다.라고 판단된다. 한편 한국경제가 기대감 감에 영향을 미치고 있다."
    en_example_to_be_summarized = "Commercial shipment is expected to grow 4.3% year over year to 138 million and witness a CAGR of 0.8% between 2025 and 2029 to hit 142.6 million. HPQ & AAPL’s Earnings Estimate Revision Goes South The Zacks Consensus Estimate for HPQ’s fiscal 2026 earnings is pegged at $3.39 per share, down 1.7% over the past 30 days, indicating a 0.3% increase over fiscal 2025’s reported figure. Quote The consensus mark for a HPQ earnings estimate has declined 0.8% to $7.12 per share over the last 30 days suggesting 5.48% growth over the fiscal 2024, suggesting a 1.7% increase in the first quarter of calendar 2025, per Canalys. Commercial shipments are expected to hit 12.8 million units in 2026, a jump of 1.7% from 2024 and a year-over-year shipment growth of 2.1% in 2025. The availability of Apple Intelligence globally with macOS Sequoia 15.4 updates in new languages, including French, German, Italian, Portuguese (Brazil), Spanish, Japanese, Korean, and Chinese (simplified), as well as localized English for Singapore and India, bodes well for Mac’s prospects. The Case for HP Stock Growing interest in Generative AI-enabled PCs might give a fresh boost to HP’s PC sales. However, HP faces meaningful downside risk if the U.S.-China tariff war escalates. The company relies heavily on China for manufacturing and assembling many of its PCs, printers, and related components. Higher import tariffs on Chinese-made goods would raise HP's production costs, forcing the company to either absorb the margin pressure or pass on costs to consumers, both negative outcomes. The growing demand for artificial intelligence (AI) powered PCs and Microsoft Windows 10 end-of-service in October 2025 are key catalysts. Gartner expects AI PCs (PC with an embedded neural processing unit) global shipments to hit 114 million units between 2026 and 2028. Commercial shipment grew 4.3% from 2025 to 2026 to hit 14.2 million units. Meanwhile, consumer PC shipments were expected to remain flat between 2022 and 2023. Commercial PC shipment growth was expected to be 1.7% in 2022 to hit 422.6 million in 2024. Meanwhile in 2023, the shipment growth grew 1.7% to hit 71.7 million. Meanwhile the consumer PC shipment was predicted to grow 1.9% in 2021 to hit 132.6 million units, an increase of 1.1% from the previous year. The increasing demand for AI-powered PCs is a key catalyst for the company’s growth."
    print(len(ko_example_to_be_summarized))
    print(len(en_example_to_be_summarized))

    # IMAGE_PATH = "C:/GD_GIT/GD_Crawling/Crawling_FnGuide/images/chart/6a7b9f07_c_4_32.png"
    # IMAGE_PATH = "C:/GD_GIT/GD_Crawling/Crawling_FnGuide/images/figure/3d5faac9_f_6_55.png"
    # PROMPT = "이 이미지에 대해 자세히 설명해줘."
    # ans_VLM = get_VLM_response(IMAGE_PATH, PROMPT)
    # print(ans_VLM)

    MODELS = [
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-8B-AWQ",
        "Qwen/Qwen3-14B-FP8",
        "Qwen/Qwen3-30B-A3B-FP8",
            ]
    LOAD_TYPES = ["fp8", "bf16", "4bit", "8bit"]

    for model_id in MODELS:
        for load_type in LOAD_TYPES:
            prompt = f"요약해줘\n{ko_example_to_be_summarized}\n요약:"
            ans_LLM = get_LLM_response(prompt, model_id=model_id, load_type=load_type, stream=True, max_new_tokens=len(prompt))
            print(f'{model_id} | {load_type} | stream=True | {time.time() - stt_time:.2f}s 소요')
            print(len(ans_LLM))
            print(f'{ans_LLM}')

            ans_LLM = get_LLM_response(prompt, model_id=model_id, load_type=load_type, stream=False, max_new_tokens=len(prompt))
            print(f'{model_id} | {load_type} | stream=False | {time.time() - stt_time:.2f}s 소요')
            print(len(ans_LLM))
            print(f'{ans_LLM}')