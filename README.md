# Thai Lexical Semantic V1

Baseline สำหรับค้นหาคำภาษาไทยที่มีความหมายใกล้เคียงกันจาก **คำศัพท์ + ความหมายในพจนานุกรมเท่านั้น** โดยยังไม่ใช้ LLM, external embedding API, corpus ภายนอก หรือ GPU

## สถานะข้อมูลปัจจุบัน

จาก \`thai_word.json\` บน \`main\`:

- 39,193 records
- 39,193 headwords ไม่ซ้ำ
- schema ปัจจุบันมีเพียง \`headword_ID\` และ \`headword_text\`
- ยังไม่มีฟิลด์ความหมาย จึงยังสร้าง semantic index จริงไม่ได้

ตัว engine จะไม่สร้างผลลัพธ์ปลอมจากชื่อคำเพียงอย่างเดียว และจะหยุดพร้อมแจ้ง schema ที่ขาด

## Schema ที่ V1 ต้องการ

ตัวอย่างขั้นต่ำ:

\`\`\`json
[
  {
    "headword_ID": 27109,
    "headword_text": "ฝน",
    "definition": "น้ำที่ตกลงมาจากเมฆเป็นเม็ด ๆ"
  },
  {
    "headword_ID": 27110,
    "headword_text": "ฝนพรำ",
    "definition": "ฝนที่ตกเป็นเม็ดเล็ก ๆ เรื่อย ๆ"
  }
]
\`\`\`

ถ้าข้อมูลจริงใช้ชื่อฟิลด์อื่น ไม่จำเป็นต้องแก้ไฟล์ สามารถส่ง \`--word-field\` และ \`--definition-field\` ให้ CLI ได้

ค่า definition รองรับ string, list หรือ nested object และจะถูกรวมเป็นข้อความก่อนประมวลผล

## V1 ทำอะไร

1. Unicode/whitespace normalization
2. ตัดคำไทยด้วย PyThaiNLP \`newmm\`
3. ใช้ headwords ทั้งชุดเป็น custom dictionary เพื่อรักษาคำเฉพาะ/คำประสม
4. ตัด Thai stopwords
5. สร้าง word + bigram TF-IDF จากความหมาย
6. สร้าง direct lexical references เมื่อความหมายกล่าวถึง headword อื่น
7. จัดอันดับจาก:
   - definition cosine similarity 55%
   - direct definition reference 25%
   - shared definition tokens 15%
   - word-form similarity 5%

น้ำหนักทั้งหมดเป็น baseline และตั้งใจให้ปรับจาก evaluation set ภายหลัง

## ติดตั้ง

\`\`\`bash
pip install -r requirements.txt
\`\`\`

ใช้ CPU เท่านั้น ไม่ต้องมี GPU

## 1) ตรวจข้อมูลก่อน

\`\`\`bash
python scripts/inspect_dictionary.py thai_word.json
\`\`\`

กับข้อมูลปัจจุบันจะรายงานว่า definition coverage = 0 และหยุดก่อน build

ถ้าความหมายอยู่ในฟิลด์ชื่อ \`meaning\`:

\`\`\`bash
python scripts/inspect_dictionary.py thai_word.json --definition-field meaning
\`\`\`

## 2) สร้าง index

\`\`\`bash
python scripts/build_index.py thai_word.json \
  --definition-field definition \
  --output artifacts/v1
\`\`\`

Artifacts:

\`\`\`text
artifacts/v1/
├── entries.json
├── metadata.json
├── references.joblib
├── tfidf_matrix.npz
├── token_sets.joblib
├── vectorizer.joblib
└── word_to_index.json
\`\`\`

## 3) ทดลองค้นหา

\`\`\`bash
python scripts/search.py "ฝน" --index artifacts/v1 --top-k 20
\`\`\`

ผลลัพธ์แต่ละคำจะมีคะแนนรวม, relation hint และคะแนนย่อย เพื่อให้ตรวจได้ว่าคำนั้นขึ้นมาเพราะอะไร

## Google Colab

V1 นี้เหมาะกับ Colab Free เพราะใช้ CPU และ sparse matrix

\`\`\`python
!git clone -b feat/dictionary-semantic-v1 https://github.com/SealNM/thai_word.git
%cd thai_word
!pip install -r requirements.txt
!python scripts/inspect_dictionary.py thai_word.json
\`\`\`

เมื่อมี definition แล้วจึงรัน build และ search ต่อ

## ทรัพยากรเป้าหมาย

สำหรับ ~50,000 entries:

- CPU: 2–4 vCPU ก็เริ่มได้
- RAM: 4–8 GB เป้าหมายเริ่มต้น
- GPU: ไม่ใช้
- index: sparse TF-IDF จึงควรอยู่ในระดับจัดการได้บน Colab Free

ตัวเลขจริงจะวัดอีกครั้งหลังได้ความยาวและโครงสร้าง definition จริง
