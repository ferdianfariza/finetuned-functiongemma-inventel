from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import torch

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI()

# ── Model config ─────────────────────────────────────────────────────────────
MODEL_ID = "ferdiannf/functiongemma-270m-it-finetuned-inventel"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # pakai float16 kalau ada GPU
    device_map="auto",
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("Model ready.")

# ── Tools schema (semua 7 fungsi) ────────────────────────────────────────────
TOOLS = [
    {
        "function": {
            "name": "getItemInfo",
            "description": "Mencari informasi detail sebuah item di gudang berdasarkan keyword nama, kode model, atau kategori. Menampilkan status unit, kondisi, lokasi, dan siapa yang sedang meminjam.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "keyword": {
                        "type": "STRING",
                        "description": "Kata kunci nama, kode model, atau kategori item yang dicari"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "function": {
            "name": "getAvailableItems",
            "description": "Mencari item yang sedang tersedia (status available) di gudang. Bisa difilter berdasarkan keyword nama atau kategori, atau tampilkan semua jika keyword kosong.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "keyword": {
                        "type": "STRING",
                        "description": "Kata kunci nama atau kategori item. Kosongkan untuk tampilkan semua item tersedia."
                    }
                },
                "required": []
            }
        }
    },
    {
        "function": {
            "name": "getMostBorrowedItems",
            "description": "Menampilkan daftar item yang paling sering dipinjam, diurutkan dari yang terbanyak.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "limit": {
                        "type": "INTEGER",
                        "description": "Jumlah item yang ingin ditampilkan"
                    }
                },
                "required": ["limit"]
            }
        }
    },
    {
        "function": {
            "name": "getUserActiveLoans",
            "description": "Menampilkan daftar pinjaman aktif milik seorang user berdasarkan userId.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "userId": {
                        "type": "STRING",
                        "description": "ID unik user yang ingin dicek pinjamannya"
                    }
                },
                "required": ["userId"]
            }
        }
    },
    {
        "function": {
            "name": "getItemLocation",
            "description": "Mencari lokasi fisik sebuah item di gudang berdasarkan nama item atau kode unit.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "keyword": {
                        "type": "STRING",
                        "description": "Nama item atau kode unit yang ingin dicari lokasinya"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "function": {
            "name": "getItemStock",
            "description": "Mengecek stok dan ringkasan jumlah unit per status (available, borrowed, dll).",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "keyword": {
                        "type": "STRING",
                        "description": "Nama item atau kategori yang ingin dicek stoknya"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "function": {
            "name": "getItemHistoryLocation",
            "description": "Melihat riwayat perpindahan lokasi sebuah item atau unit, termasuk siapa yang memindahkan dan kapan.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "keyword": {
                        "type": "STRING",
                        "description": "Nama item atau kode unit yang ingin dilihat riwayat lokasinya"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
]

# ── Request schema ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    message: str
    userId: str = None  # opsional, dikirim dari bot kalau user sudah login

# ── Parser output model ───────────────────────────────────────────────────────
def parse_function_call(output: str):
    """
    Parse output model format FunctionGemma:
    <start_function_call>call:funcName{key:<escape>value<escape>}<end_function_call>
    """
    call_pattern = r"<start_function_call>(.*?)<end_function_call>"
    raw_calls = re.findall(call_pattern, output, re.DOTALL)

    results = []
    for raw_call in raw_calls:
        if not raw_call.strip().startswith("call:"):
            continue
        try:
            pre_brace, args_segment = raw_call.split("{", 1)
            function_name = pre_brace.replace("call:", "").strip()
            args_content = args_segment.strip().rstrip("}")
            arg_pattern = r"(?P<key>[^:,]*?):<escape>(?P<value>.*?)<escape>"
            arg_matches = re.finditer(arg_pattern, args_content, re.DOTALL)
            arguments = {m.group("key").strip(): m.group("value") for m in arg_matches}
            results.append({"name": function_name, "arguments": arguments})
        except ValueError:
            continue

    return results

# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(body: PredictRequest):
    messages = [
        {
            "role": "developer",
            "content": "You are an inventory warehouse assistant. Answer user questions by calling the appropriate function. Only call a function when needed."
        },
        {
            "role": "user",
            "content": body.message
        }
    ]

    # Inject userId ke pesan kalau tersedia
    # Berguna supaya model tau konteks siapa yang bertanya
    if body.userId:
        messages[0]["content"] += f" Current user ID is: {body.userId}."

    prompt = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True
    )

    output = pipe(
        prompt,
        max_new_tokens=200,
        temperature=0.001,
        do_sample=False,
    )[0]["generated_text"]

    raw_output = output[len(prompt):].strip()
    function_calls = parse_function_call(raw_output)

    if function_calls:
        return {
            "type": "function_call",
            "calls": function_calls
        }
    else:
        # NO_FUNCTION_CALL — model jawab teks biasa
        clean_text = raw_output.replace("<end_of_turn>", "").strip()
        return {
            "type": "text",
            "text": clean_text
        }

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}