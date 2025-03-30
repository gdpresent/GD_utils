from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ko_tokenizer = AutoTokenizer.from_pretrained("psyche/KoT5-summarization")
ko_model = AutoModelForSeq2SeqLM.from_pretrained("psyche/KoT5-summarization")
en_tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
en_model = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-tglobal-base")
def get_summary(ori_text, lang="kr", max_pct_of_ori=50, min_pct_of_ori=10):
    if "kr" in lang.lower():
        tokenizer = ko_tokenizer
        model = ko_model
    elif "en" in lang.lower():
        tokenizer = en_tokenizer
        model = en_model
    else:
        raise ValueError("lang should be 'ko' or 'en'")
    inputs = tokenizer(ori_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(**inputs,
                                 max_length=int(len(article_text) * max_pct_of_ori // 100),
                                 min_length=int(len(article_text) * min_pct_of_ori // 100),
                                 num_beams=5,
                                 no_repeat_ngram_size=3,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary