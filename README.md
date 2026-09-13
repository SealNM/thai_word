# Thai Lexical Semantic V1

Baseline สำหรับค้นหาคำภาษาไทยที่มีความหมายใกล้เคียงกันจาก **คำศัพท์ + ความหมายในพจนานุกรมเท่านั้น** โดยยังไม่ใช้ LLM, external embedding API, corpus ภายนอก หรือ GPU

## Schema จริงที่ V1 ใช้เป็นค่าเริ่มต้น

จากตัวอย่างข้อมูลพจนานุกรมจริง:

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

## คำเดียวหลายความหมาย

พจนานุกรมจริงมีหลาย records ที่ใช้ `headword_text` เดียวกันแต่คนละ `word_ID` / คนละความหมาย เช่น `พรำ`

V1 จะ:

1. รวม records ที่มี headword เดียวกันเป็น semantic entry เดียว
2. เก็บทุกความหมายที่ไม่ซ้ำ
3. เก็บ `source_ids` ของทุก record
4. เก็บ `sense_count`
5. ใช้ความหมายทั้งหมดร่วมกันในการสร้าง representation

จึงไม่ทิ้ง sense อื่นของคำเดียวกัน

## สถานะ `thai_word.json` เดิมใน repo

ไฟล์ `thai_word.json` บน `main` มี 39,193 records / 39,193 headwords ไม่ซ้ำ แต่มีเพียง `headword_ID` และ `headword_text` จึงยังใช้สร้าง semantic index ไม่ได้

สำหรับการทดลอง V1 สาขานี้มี `thai_dictionary_test.json` ที่มี `definition_text` อยู่แล้ว และ pipeline จะเลือกไฟล์นี้โดยอัตโนมัติเมื่อไม่ส่ง path

## V1 ทำอะไร

1. Unicode/whitespace normalization
2. ตัดคำไทยด้วย PyThaiNLP `newmm`
3. ใช้ headwords ทั้งชุดเป็น custom dictionary เพื่อรักษาคำเฉพาะ/คำประสม
4. ตัด Thai stopwords
5. รวม multiple senses ของ headword เดียวกัน
6. สร้าง word + bigram TF-IDF จากความหมาย
7. สร้าง direct lexical references เมื่อความหมายกล่าวถึง headword อื่น
8. จัดอันดับจาก:
   - definition cosine similarity 55%
   - direct definition reference 25%
   - shared definition tokens 15%
   - word-form similarity 5%

น้ำหนักทั้งหมดเป็น baseline และตั้งใจให้ปรับจาก evaluation set ภายหลัง

## ติดตั้ง

```bash
pip install -r requirements.txt
```

ใช้ CPU เท่านั้น ไม่ต้องมี GPU

## 1) ตรวจไฟล์พจนานุกรม

สาขานี้มีไฟล์ `thai_dictionary_test.json` อยู่แล้ว และถูกตั้งเป็นค่าเริ่มต้นของ pipeline ดังนั้นรันได้ทันที:

```bash
python scripts/inspect_dictionary.py
```

ถ้าต้องการใช้ไฟล์อื่น ยังส่ง path เองได้:

```bash
python scripts/inspect_dictionary.py another.json
```

ตัว inspector จะรายงาน:

- จำนวน records
- จำนวน ID
- จำนวน headwords
- จำนวนคำไม่ซ้ำ
- จำนวน duplicate headword rows
- definition coverage
- schema fields ที่พบ

ถ้า definition coverage เป็น 0 ระบบจะหยุดก่อน build เพื่อไม่สร้าง semantic index จากชื่อคำเพียงอย่างเดียว

### Dataset ที่ใช้ชื่อฟิลด์อื่น

```bash
python scripts/inspect_dictionary.py another.json \
  --id-field id \
  --word-field word \
  --definition-field meaning
```

## 2) สร้าง index

สำหรับ schema จริง `word_ID / headword_text / definition_text` ไม่ต้องระบุ field เพิ่ม:

```bash
python scripts/build_index.py --output artifacts/v1
```

Artifacts:

```text
artifacts/v1/
├── entries.json
├── metadata.json
├── references.joblib
├── tfidf_matrix.npz
├── token_sets.joblib
├── vectorizer.joblib
└── word_to_index.json
```

`metadata.json` จะมีทั้ง `entries_indexed` และ `senses_indexed` เพื่อแยกจำนวน headwords หลัง merge ออกจากจำนวนความหมายที่ใช้จริง

## 3) ทดลองค้นหา

```bash
python scripts/search.py "ฝน" --index artifacts/v1 --top-k 20
```

หรือ:

```bash
python scripts/search.py "พรำ" --index artifacts/v1 --top-k 20
```

ผลลัพธ์แต่ละคำมี:

- score
- relation hint
- definition cosine
- direct reference
- shared tokens
- word-form score
- definition ที่รวมแล้ว
- sense_count
- source_ids

จึงตรวจสอบได้ว่าคำนั้นขึ้นมาเพราะอะไร

## Google Colab

V1 เหมาะกับ Colab Free เพราะใช้ CPU และ sparse matrix

### Cell 1 — clone branch

```python
!git clone -b feat/dictionary-semantic-v1 https://github.com/SealNM/thai_word.git
%cd thai_word
```

### Cell 2 — dependencies

```python
!pip install -r requirements.txt
```

### Cell 3 — inspect

ไฟล์ `thai_dictionary_test.json` อยู่ใน branch แล้ว ไม่ต้อง upload เพิ่ม:

```python
!python scripts/inspect_dictionary.py
```

### Cell 4 — build

```python
!python scripts/build_index.py --output artifacts/v1
```

### Cell 5 — search

```python
!python scripts/search.py "พรำ" --index artifacts/v1 --top-k 20
```

## ทรัพยากรเป้าหมาย

สำหรับประมาณ 50,000 records:

- CPU: 2–4 vCPU ก็เริ่มได้
- RAM: 4–8 GB เป้าหมายเริ่มต้น
- GPU: ไม่ใช้
- index: sparse TF-IDF

ตัวเลขจริงจะวัดจากไฟล์เต็มหลัง build ครั้งแรก
