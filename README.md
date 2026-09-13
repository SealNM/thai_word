# Thai Lexical Semantic V2

V2 ต่อจาก V1 โดยเก็บ lexical/sparse baseline เดิมไว้ทั้งหมด แล้วเพิ่ม **sense-level dense embeddings** เพื่อช่วยค้นคำที่ความหมายใกล้กันแม้นิยามใช้คนละถ้อยคำ โดยยังไม่ใช้ LLM หรือ external embedding API และยังออกแบบให้รันบน Google Colab ได้

V1 ยังคงเป็น baseline ที่สำคัญสำหรับ direct dictionary relation, sense-aware TF-IDF และ lexical tiers ส่วน V2 เพิ่ม dense retrieval และรวมอันดับด้วย weighted Reciprocal Rank Fusion (RRF)

## V2 architecture

```text
query + selected sense
        │
        ├── V1 lexical graph + TF-IDF
        │
        └── dense embedding search (sense-level)
                    │
                    ▼
             candidate union
                    │
                    ▼
          weighted RRF fusion
                    │
                    ▼
              final ranking
```

หลักสำคัญ:

- dense embedding สร้าง **ต่อ sense** ไม่ใช่ต่อ headword
- direct lexical relations Tier 4–5 แบบ standalone ถูกปกป้องไว้เหนือ dense-only candidates
- Tier 0–1 ไม่ได้ lexical-rank bonus ใน fusion เพื่อเปิดทางให้ dense semantic ช่วยแก้กรณีเช่น `พูด → เอ่ย`
- Tier 2 ได้ lexical weight 0.25, Tier 3 ได้ 0.5, Tier 4–5 ได้ 1.0
- ใช้ RRF เพื่อไม่ต้องเอา cosine ของ E5 กับ GTE ซึ่งมีสเกลต่างกันมาบวกตรง ๆ
- V1 artifacts ไม่ต้อง rebuild เมื่อเปลี่ยน dense model

Built-in dense models:

| key | model | หมายเหตุ |
| --- | --- | --- |
| `e5-small` | `intfloat/multilingual-e5-small` | baseline เบา, 384 dimensions, symmetric `query:` prefix |
| `e5-base` | `intfloat/multilingual-e5-base` | stable challenger, 768 dimensions, standard XLM-R backbone |
| `gte-base-experimental` | `Alibaba-NLP/gte-multilingual-base` | experimental only; uses custom remote code |

E5 ใช้ `query:` ทั้ง query และ dictionary sense เพราะงานนี้เป็น semantic similarity / paraphrase-style retrieval มากกว่า asymmetric passage QA

## Google Colab — เริ่มใหม่ทั้งหมดจากศูนย์

ส่วนนี้เป็นเส้นทางหลักที่แนะนำสำหรับ Colab Free หากต้องการทดสอบ V2 ใหม่ทั้งหมดให้ทำตามลำดับนี้โดยไม่ข้ามขั้น

> แนะนำ: ถ้า Colab Free ให้ GPU ได้ ให้เลือก **Runtime → Change runtime type → T4 GPU** ก่อนเริ่ม Step 1 เพราะการ build dense index บน CPU ทำได้แต่ช้ากว่ามาก

### Step 1 — ล้าง repo เก่าและ clone V2 ใหม่

เริ่มจาก directory `/content` เสมอ เพื่อป้องกันปัญหา clone ซ้อนเป็น `/content/thai_word/thai_word`

```python
%cd /content
!rm -rf /content/thai_word
!git clone -b feat/dictionary-semantic-v2 https://github.com/SealNM/thai_word.git
%cd /content/thai_word

!git rev-parse --show-toplevel
!git log -1 --oneline
```

ผลของ `git rev-parse --show-toplevel` ควรเป็น:

```text
/content/thai_word
```

### Step 2 — ติดตั้ง dependencies

```python
!pip install -r requirements.txt
```

warning เรื่อง Hugging Face `HF_TOKEN` ไม่ใช่ error สำหรับ public models ที่เราใช้ หากไม่ได้ตั้ง token ยังดาวน์โหลดได้ เพียงแต่ rate limit ต่ำกว่า

### Step 3 — ตรวจ environment และ GPU

```python
import torch

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

ถ้าใช้ GPU runtime ควรเห็น:

```text
cuda available: True
```

หากเลือก GPU ใน Colab แล้วแต่ยังได้ `False` ให้ restart runtime แล้วเริ่ม Step 1 ใหม่ ไม่แนะนำให้สุ่มติดตั้ง PyTorch คนละ build ทับ environment

ถ้า Colab ไม่ให้ GPU สามารถทำต่อด้วย CPU ได้ ระบบจะ fallback จาก `--device cuda` เป็น CPU โดยอัตโนมัติ แต่การ build dense index จะช้ากว่า

### Step 4 — รัน unit tests ก่อน build

```python
!python -m unittest discover -s tests
```

ควรให้ tests ผ่านก่อนดำเนินการต่อ หากขั้นนี้แดงให้แก้ก่อน build index เพื่อไม่เสียเวลาสร้าง embeddings ใหม่

### Step 5 — ตรวจพจนานุกรม

```python
!python scripts/inspect_dictionary.py
```

ตรวจว่าไฟล์พจนานุกรมอ่านได้และ `definition_coverage` ไม่เป็นศูนย์

### Step 6 — สร้าง V1 lexical index ใหม่

รอบเริ่มใหม่ให้ลบ artifact เดิมทั้งหมดก่อน:

```python
!rm -rf artifacts/v1
!python scripts/build_index.py --output artifacts/v1
```

จากนั้นทดลองดู senses และค้น V1:

```python
!python scripts/search.py "ฝน" --index artifacts/v1 --list-senses
!python scripts/search.py "ฝน" --index artifacts/v1 --top-k 10
```

ถ้าสองคำสั่งนี้ทำงานจึงค่อยไป dense index

### Step 7 — ล้าง V2 artifacts เก่า

```python
!rm -rf artifacts/v2
!mkdir -p artifacts/v2
```

### Step 8 — Build E5-small baseline

ถ้า GPU พร้อม:

```python
!python scripts/build_dense_index.py \
  --index artifacts/v1 \
  --model e5-small \
  --output artifacts/v2/e5-small \
  --device cuda \
  --batch-size 64
```

ถ้าใช้ CPU:

```python
!python scripts/build_dense_index.py \
  --index artifacts/v1 \
  --model e5-small \
  --output artifacts/v2/e5-small \
  --device cpu \
  --batch-size 32
```

เมื่อสำเร็จควรมี:

```text
artifacts/v2/e5-small/dense_embeddings.npy
artifacts/v2/e5-small/dense_metadata.json
```

ตรวจ metadata:

```python
!cat artifacts/v2/e5-small/dense_metadata.json
```

### Step 9 — ทดสอบ hybrid search ด้วย E5-small

```python
!python scripts/search_v2.py "พูด" \
  --index artifacts/v1 \
  --dense-index artifacts/v2/e5-small \
  --top-k 20 \
  --device cuda
```

ถ้าไม่มี GPU เปลี่ยนเป็น `--device cpu`

ตัวอย่าง query หลายความหมาย:

```python
!python scripts/search_v2.py "รัก" \
  --index artifacts/v1 \
  --dense-index artifacts/v2/e5-small \
  --sense 3 \
  --top-k 20 \
  --device cuda
```

### Step 10 — Build E5-base challenger

E5-base เป็น challenger หลักของ V2 และใช้ 768 dimensions

GPU:

```python
!python scripts/build_dense_index.py \
  --index artifacts/v1 \
  --model e5-base \
  --output artifacts/v2/e5-base \
  --device cuda \
  --batch-size 32
```

CPU:

```python
!python scripts/build_dense_index.py \
  --index artifacts/v1 \
  --model e5-base \
  --output artifacts/v2/e5-base \
  --device cpu \
  --batch-size 16
```

### Step 11 — Benchmark V1 vs E5-small vs E5-base

GPU:

```python
!python scripts/evaluate_v2.py \
  --index artifacts/v1 \
  --dense-index artifacts/v2/e5-small \
  --dense-index artifacts/v2/e5-base \
  --top-k 10 \
  --device cuda \
  --output evaluation/v2_report.json
```

CPU:

```python
!python scripts/evaluate_v2.py \
  --index artifacts/v1 \
  --dense-index artifacts/v2/e5-small \
  --dense-index artifacts/v2/e5-base \
  --top-k 10 \
  --device cpu \
  --output evaluation/v2_report.json
```

benchmark โหลด dense model ทีละตัวและเคลียร์ GPU ก่อนโหลดตัวถัดไป เพื่อลด peak RAM/VRAM บน Colab Free

evaluation set pin ความหมายของคำกำกวมไว้แล้ว เช่น `รัก = sense 3`, `สวย = sense 1`, `มืด = sense 1`, `บ้าน = sense 1`

### Step 12 — เก็บผล benchmark

ผลฉบับเต็มอยู่ที่:

```text
evaluation/v2_report.json
```

metadata ของแต่ละ dense model จะบันทึก:
- model id
- dimensions
- embedding size
- model load time
- encode time
- requested device
- device ที่ใช้จริง

ใช้ข้อมูลเหล่านี้ร่วมกับคุณภาพผลค้นหาเพื่อตัดสินว่าจะใช้ E5-small หรือ E5-base เป็น default

### Experimental: GTE

`Alibaba-NLP/gte-multilingual-base` ไม่อยู่ใน benchmark หลักอีกแล้ว เพราะพึ่ง Hugging Face custom remote modeling code และพบ compatibility issue กับ PyTorch/Transformers รุ่นปัจจุบันบน Colab

หากต้องการทดลองโดยตั้งใจยังเรียกได้ด้วย:

```text
--model gte-base-experimental
```

แต่ไม่ควรใช้เป็นเส้นทางมาตรฐานของ V2 ในตอนนี้

## V2 artifacts

แต่ละ dense model เก็บแยก directory:

```text
artifacts/v2/e5-small/
├── dense_embeddings.npy
└── dense_metadata.json

artifacts/v2/e5-base/
├── dense_embeddings.npy
└── dense_metadata.json
```

ตัว embedding เป็น normalized float32 และเรียงแถวตรงกับ `artifacts/v1/senses.json`

## Schema จริงที่ V1 ใช้เป็นค่าเริ่มต้น

จากข้อมูลพจนานุกรมจริง:

```json
[
  {
    "word_ID": 27959,
    "headword_text": "พรำ",
    "definition_text": "ตกน้อย ๆ เรื่อยไป (ใช้แก่ฝน) ในคำว่า ฝนพรำ."
  },
  {
    "word_ID": 27960,
    "headword_text": "พรำ",
    "definition_text": "อาการที่ฝนตกน้อย ๆ เรื่อยไป ใช้ว่า ฝนตกพรำ ฝนตกพรำ ๆ."
  }
]
```

ค่า default ของ pipeline คือ:

- ID: `word_ID`
- คำ: `headword_text`
- ความหมาย: `definition_text`

ถ้า dataset รุ่นอื่นใช้ชื่อฟิลด์ต่างออกไป ยัง override ได้ด้วย `--id-field`, `--word-field` และ `--definition-field`

## คำเดียวหลายความหมาย: sense-aware retrieval

พจนานุกรมจริงมีหลาย records ที่ใช้ `headword_text` เดียวกันแต่คนละ `word_ID` / คนละความหมาย

V1 เก็บ headword เดียวเป็น entry เดียวเพื่อไม่ให้ผลลัพธ์ซ้ำ แต่ **ไม่เอาความหมายทั้งหมดมาต่อรวมเป็น vector เดียวอีกแล้ว** เพราะคำหลายความหมาย เช่น `ฝน` อาจหมายถึงทั้งฝนจากฟ้า, รอบปี หรือกริยาฝน/ลับ

ระบบจะ:

1. รวม metadata ของ headword เดียวกันไว้ด้วยกัน
2. เก็บทุก definition เป็น sense แยกกันตามลำดับเดิมของพจนานุกรม
3. สร้าง TF-IDF vector แยกต่อ sense
4. ค้นหาและให้คะแนนจากคู่ sense ที่ตรงกันที่สุด
5. ค่าเริ่มต้นของ exact-headword query ใช้ **sense 1**
6. เลือก sense อื่นได้ด้วย `--sense N`
7. แสดงทั้ง `query_sense` และ `matched_candidate_sense` ในผลลัพธ์

วิธีนี้ลดการปนกันของความหมายโดยไม่ต้องใช้โมเดลภายนอก

## V1 ทำอะไร

1. Unicode/whitespace normalization
2. ตัดคำไทยด้วย PyThaiNLP `newmm`
3. ใช้ headwords ทั้งชุดเป็น custom dictionary เพื่อรักษาคำเฉพาะ/คำประสม
4. ตัด Thai stopwords
5. เก็บ multiple senses ของ headword เดียวกันแบบแยก representation
6. สร้าง word + bigram TF-IDF จาก definition ของแต่ละ sense
7. สร้าง direct lexical references เมื่อ definition กล่าวถึง headword อื่น
8. candidate generation ใช้ทั้ง definition similarity และ direct references
9. แยกคุณภาพ lexical reference แทนการให้คะแนนทุก mention เท่ากัน:
   - นิยามตรง เช่น `ฝน.` = คำพ้อง/คำแทนโดยตรง
   - นิยามชนิดย่อย เช่น `ฝนเม็ดใหญ่...` = type-of relation
   - mention ในตัวอย่าง เช่น `เช่น เมฆอุ้มฝน` = contextual relation
   - mention เชิงเกี่ยวข้อง เช่น `เทวดาแห่งฝน` = associated relation
10. จัดอันดับ exact-headword query แบบ hierarchical relation tier:
   - Tier 5: direct gloss / synonym เช่น `ฝน.`
   - Tier 4: alternative direct gloss เช่น `เมฆ, ฝน.`
   - Tier 3: subtype / kind-of เช่น `ฝนเม็ดใหญ่...`
   - Tier 2: contextual / associated mention
   - Tier 1: คำที่ถูกกล่าวถึงในนิยามของ query
   - Tier 0: definition similarity อย่างเดียว
11. cosine / shared tokens / word-form ใช้เป็น **ตัวตัดสินภายใน tier** ไม่ให้ similarity ที่สูงกว่าข้ามชนิดความสัมพันธ์ที่แข็งแรงกว่าได้
12. headword ที่ขึ้นต้นหรือลงท้ายด้วย `-` ถือเป็น bound form และลดลง 1 tier แทนการลบทิ้ง เพื่อยังคงค้นเจอข้อมูลทางศัพท์ได้

ตัวอย่าง: `พิรุณ = ฝน.` จะต้องอยู่เหนือ `พลาหก = เมฆ, ฝน.` เสมอ แม้ `พลาหก` จะมี cosine/shared tokens สูงกว่า และ `พรรษ-` จะต่ำกว่า `พรรษ` ในฐานะรูปประกอบคำ

สำหรับ query ที่ไม่ตรง headword ในพจนานุกรม ระบบยัง fallback ไปใช้ semantic similarity จาก definition เป็นหลัก

น้ำหนักและ tier เหล่านี้ยังเป็น baseline และตั้งใจให้ปรับจาก evaluation set ภายหลัง

## ติดตั้ง

```bash
pip install -r requirements.txt
```

V1 ใช้ CPU ได้ทั้งหมด ส่วน V2 dense embedding ใช้ CPU ได้เช่นกันแต่แนะนำ GPU เมื่อ build index

## 1) ตรวจไฟล์พจนานุกรม

```bash
python scripts/inspect_dictionary.py
```

ถ้าต้องการใช้ไฟล์อื่น:

```bash
python scripts/inspect_dictionary.py another.json
```

## 2) สร้าง index

ถ้าเคย build artifact schema รุ่นก่อนหน้า ให้ลบแล้วสร้างใหม่:

```bash
rm -rf artifacts/v1
python scripts/build_index.py --output artifacts/v1
```

Artifacts:

```text
artifacts/v1/
├── entries.json
├── senses.json
├── metadata.json
├── entry_to_senses.joblib
├── references.joblib
├── reverse_references.joblib
├── tfidf_matrix.npz
├── token_sets.joblib
├── vectorizer.joblib
└── word_to_index.json
```

`metadata.json` จะมี `entries_indexed`, `senses_indexed` และ `representation: "sense_level"`

## 3) ตรวจว่าคำหนึ่งมีความหมายอะไรบ้าง

ตัวอย่างคำว่า `ฝน`:

```bash
python scripts/search.py "ฝน" --index artifacts/v1 --list-senses
```

ผลจะคืนลำดับ sense และ definition ตามพจนานุกรม

## 4) ทดลองค้นหา

ค่าเริ่มต้น exact headword จะใช้ sense 1:

```bash
python scripts/search.py "ฝน" --index artifacts/v1 --top-k 20
```

เลือกความหมายอื่นได้ เช่น sense 2:

```bash
python scripts/search.py "ฝน" --index artifacts/v1 --sense 2 --top-k 20
```

หรือค้นคำอื่น:

```bash
python scripts/search.py "พรำ" --index artifacts/v1 --top-k 20
```

ผลลัพธ์แต่ละคำมี:

- score
- relation hint
- relation tier
- lexical form (`standalone` / `bound_form`)
- definition cosine
- reverse reference strength
- forward reference strength
- shared tokens
- word-form score
- `query_sense`
- `matched_candidate_sense`
- definition ที่ใช้จับคู่จริง
- sense_count
- source_ids

## ทดสอบ logic ของ sense selection

```bash
python -m unittest discover -s tests
```

test นี้ไม่ต้องโหลดพจนานุกรมเต็ม และตรวจว่า default search ใช้ sense 1 ขณะที่ `--sense 2` เปลี่ยน semantic neighborhood ได้จริง

## Evaluation หลายกลุ่มคำ

หลังจากคำว่า `ฝน` ผ่านเกณฑ์เชิงโครงสร้างแล้ว ไม่ควรจูนต่อจากคำเดียว เพราะเสี่ยง overfit

สาขานี้มีชุดประเมินเล็ก ๆ ที่ครอบคลุม noun / verb / adjective / emotion:

```text
evaluation/v1_queries.json
```

รันแบบสรุปก่อนเพื่อดูครบทุก query โดยไม่ให้ output ยาวเกินไป:

```bash
python scripts/evaluate.py --index artifacts/v1 --top-k 10 --summary
```

ถ้าต้องการ full JSON ให้รันโดยไม่ใส่ `--summary`:

```bash
python scripts/evaluate.py --index artifacts/v1 --top-k 10
```

ถ้าต้องการบันทึกผลเป็น JSON เพื่อเปรียบเทียบก่อน/หลังการแก้ ranking:

```bash
python scripts/evaluate.py \
  --index artifacts/v1 \
  --top-k 10 \
  --output evaluation/latest_report.json
```

รายงานจะแสดงต่อ query:

- category
- requested / selected sense
- senses ทั้งหมดที่พจนานุกรมมี
- จำนวนผลลัพธ์แยกตาม relation tier
- top results พร้อม definition, relation hint, lexical form และ signals

สำหรับ query ที่ `sense: null` และมีหลาย raw senses โหมด `--summary` จะแสดงคำเตือนพร้อมรายการ sense ทั้งหมดก่อน ไม่ควรตัดสินคุณภาพ ranking จาก sense แรกโดยอัตโนมัติ ให้กำหนดหมายเลข sense ที่ตรงกับความหมายเป้าหมายใน `evaluation/v1_queries.json` แล้วรันใหม่

เมื่อระบุ `--sense N` หรือ `sense: N` ระบบถือว่าเป็น **strict sense search**:
- direct gloss ที่ระบุเพียง headword เช่น `รัก.` แต่ไม่มีหลักฐานใน definition ว่าชี้ไปยัง sense ใด จะยังแสดงได้ แต่ติด `sense_resolution: headword_reference_ambiguous`
- reference แบบนี้จะถูกลด tier เพื่อไม่ให้ข้ามมาครองผลลัพธ์ของ homonym ที่ผิดความหมาย
- ถ้า candidate entry มี definition อื่นที่สนับสนุน sense ที่เลือก ระบบใช้ `entry_semantic_cosine` เป็นหลักฐานประกอบได้

นอกจากนี้ forward reference จากนิยาม query แยก gloss ออกจากตัวอย่างแล้ว เช่น `ไว เช่น กินเร็ว` จะให้น้ำหนัก `ไว` สูง แต่ `กิน` ต่ำ และ `ไม่ชักช้า` จะไม่ถูกนับเป็นคำพ้องของ `เร็ว`

เป้าหมายของรอบนี้คือดูว่า ranking architecture ใช้ได้ทั่วไป ไม่ใช่บังคับให้ทุก query มีคำตอบตามรายการที่เขียนไว้ล่วงหน้า

## ทรัพยากรเป้าหมาย

สำหรับพจนานุกรมระดับหลายหมื่น records:

- V1 lexical: CPU-only, 2–4 vCPU ใช้งานได้
- V1 RAM: 4–8 GB เป็นเป้าหมายเริ่มต้น
- V2 dense build: GPU แนะนำแต่ไม่บังคับ
- E5-small เบากว่าและเป็น baseline แรกสำหรับ Colab
- E5-base หนักกว่า E5-small จึงใช้ batch size เริ่มต้นต่ำกว่า
- dense index ใช้ normalized float32 และยังไม่ต้องใช้ FAISS ใน V2 baseline
- benchmark หลายโมเดลโหลดทีละโมเดลเพื่อลด peak RAM/VRAM
- index รวม: sparse TF-IDF + lexical graph + dense sense embeddings

ตัวเลข build time / RAM / VRAM จริงจะเก็บจาก Colab หลัง benchmark เต็มชุด
