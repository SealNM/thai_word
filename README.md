# Thai Lexical Semantic V1

Baseline สำหรับค้นหาคำภาษาไทยที่มีความหมายใกล้เคียงกันจาก **คำศัพท์ + ความหมายในพจนานุกรมเท่านั้น** โดยยังไม่ใช้ LLM, external embedding API, corpus ภายนอก หรือ GPU

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

ใช้ CPU เท่านั้น ไม่ต้องมี GPU

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

## Google Colab

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

```python
!python scripts/inspect_dictionary.py
```

### Cell 4 — build

```python
!rm -rf artifacts/v1
!python scripts/build_index.py --output artifacts/v1
```

### Cell 5 — ดู senses ของคำว่า ฝน

```python
!python scripts/search.py "ฝน" --index artifacts/v1 --list-senses
```

### Cell 6 — ค้นด้วยความหมายหลัก

```python
!python scripts/search.py "ฝน" --index artifacts/v1 --top-k 20
```

ถ้าต้องการ sense อื่น ให้เติม `--sense 2`, `--sense 3` ตามรายการที่ Cell 5 แสดง

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

สำหรับประมาณ 50,000 records:

- CPU: 2–4 vCPU ก็เริ่มได้
- RAM: 4–8 GB เป้าหมายเริ่มต้น
- GPU: ไม่ใช้
- index: sparse TF-IDF ระดับ sense

ตัวเลขจริงจะวัดจากไฟล์เต็มหลัง build
