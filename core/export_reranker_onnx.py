import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "./models/bge-reranker-base"
OUT_DIR = "reranker-onnx"
os.makedirs(OUT_DIR, exist_ok=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# dummy 输入，导出动态 batch / 动态长度
enc = tokenizer(["示例查询"], ["示例段落"], padding="max_length",
                truncation=True, max_length=128, return_tensors="pt")

def wrapped(input_ids, attention_mask, token_type_ids):
    # 只输出 logits，避免导出多余输出
    return model(input_ids=input_ids, attention_mask=attention_mask,
                 token_type_ids=token_type_ids).logits

input_names = ["input_ids", "attention_mask", "token_type_ids"]
dynamic_axes = {n: {0: "batch", 1: "seq"} for n in input_names}

with torch.no_grad():
    torch.onnx.export(
        wrapped,
        (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
        os.path.join(OUT_DIR, "model.onnx"),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )

tokenizer.save_pretrained(OUT_DIR)
print("完成 ->", OUT_DIR)
