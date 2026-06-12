# Neural Search Engine

ტექსტის მოძიების ნეირონული სისტემა (neural text retrieval system), რომელიც ახდენს ბუნებრივ ენაზე დასმული მოთხოვნებისა (queries)
და დოკუმენტის პასაჟების კოდირებას მკვრივ ვექტორულ ემბედინგებად (dense vector embeddings), ახდენს პასაჟების რანჟირებას კოსინუსური
მსგავსების (cosine similarity) მიხედვით. შექმნილია როგორც ჯგუფური პროექტი NLP (ბუნებრივი ენის დამუშავების) კურსისთვის.


## Project Structure
NLP_FINAL_PROJECT/

├── checkpoints/          # Saved model checkpoints

├── data/

│   ├── book/             # Jurafsky & Martin SLP3 chapter texts

│   ├── pairs/            # train.json, val.json, test.json

│   └── raw/              # Raw MS MARCO download

├── notebooks/

│   ├── data_exploration.ipynb

│   ├── training.ipynb

│   └── evaluation.ipynb

├── src/

│   ├── baseline.py       # TF-IDF search engine

│   ├── corpus.py         # Book text chunking (200-300 words)

│   ├── data_prep.py      # MS MARCO download and split

│   ├── evaluate.py       # Recall@k and MRR evaluation

│   ├── fetch_book.py     # Download SLP3 chapters from Stanford

│   ├── loss.py           # InfoNCE and Triplet loss

│   ├── model.py          # DistilBERT-based TextEncoder

│   ├── search.py         # NeuralSearchEngine + encode_texts

│   ├── train.py          # Training loop

│   ├── transformer_encoder.py  # Transformer built from scratch

│   └── vocab.py          # Word-level tokenizer and vocabulary

└── tests/

└── test_book_search.py


## Pipeline
```
query → text encoder → query embedding → cosine similarity → top-k chunks
```


## Key Concepts

კონტრასტული სწავლება (Contrastive Learning): ენკოდერი მუშაობს პოზიტიურ წყვილებზე (მოთხოვნა + რელევანტური პასაჟი)
და ნეგატიურ წყვილებზე (მოთხოვნა + შეუსაბამო პასაჟი). მოდელი სწავლობს, რომ ემბედინგების სივრცეში პოზიტიური წყვილები
ერთმანეთთან ახლოს მიიზიდოს, ხოლო ნეგატიური წყვილები ერთმანეთისგან მაქსიმალურად დააშოროს (განაზიდოს).

ლექსიკური შეუსაბამობა (Vocabulary Mismatch): TF-IDF ვერ უმკლავდება სიტუაციებს, როდესაც მოთხოვნასა და დოკუმენტში
ერთი და იმავე იდეის გადმოსაცემად სხვადასხვა სიტყვებია გამოყენებული (მაგალითად, „მანქანა“ და „ავტომობილი“).
მკვრივი ძიება (dense retrieval) ამ პრობლემას იმით აგვარებს, რომ პერეფრაზებს ერთმანეთთან ახლომდებარ
ე ვექტორებად ასახავს, მიუხედავად მათი ზედაპირული (სიტყვიერი) ფორმისა.

ბეჩშიდა ნეგატივები (In-batch Negatives): InfoNCE დანაკარგის ფუნქცია მიმდინარე ბეჩში (ჯგუფში) არსებულ ყველა
სხვა პოზიტიურ წყვილს განიხილავს, როგორც დამატებით ნეგატიურ მაგალითს. შედეგად, 64-იანი ბეჩის ზომის (batch size)
შემთხვევაში, თითოეული მოთხოვნა ყოველ ბიჯზე 1 ნეგატივის ნაცვლად 127 ნეგატივს ხედავს.


## Data

ჩვენ ვიყენებთ MS MARCO v2.1-ს (50,000 საძიებო მოთხოვნა–პასაჟის ტრიპლეტი):

მოთხოვნა (Query): Bing-ის საძიებო სისტემის რეალური კითხვა
პოზიტივი (Positive): ადამიანის მიერ შერჩეული რელევანტური პასაჟი
ნეგატივი (Negative): იმავე მოთხოვნის ჯგუფიდან აღებული შეუსაბამო პასაჟი

მონაცემთა გაყოფა (Split): 90% საწვრთნელი (train) / 5% სავალიდაციო (val) / 5% სატესტო (test).

დემო საძიებო კორპუსი აგებულია Dan Jurafsky & James H. Martin — Speech and Language Processing (3rd ed.) სახელმძღვანელოთი,
რომელიც დაყოფილია 200–300 სიტყვიან ტექსტურ ფრაგმენტებად (chunks).



## Models

### Transformer from Scratch
- სიტყვათა დონის ტოკენიზატორი (Word-level tokenizer), 30,000 ტოკენისგან შემდგარი ვოქებულარი
- ორფენიანი ტრანსფორმერი 4-ჰედიანი self-attention მექანიზმით, d_model=256
- Masked mean pooling + $L_2$ ნორმალიზაცია
- ~8.8 მლნ პარამეტრი, გაწვრთნილი სრულიად შემთხვევითი ინიციალიზაციიდან

### DistilBERT (Reference)
- Fine-tuned `distilbert-base-uncased`
- იგივე პულინგისა (pooling) და ნორმალიზაციის ჰედი (head)
= გამოიყენება, როგორც მოდელის შესაძლებლობების ძლიერი ზედა ზღვარი (upper-bound reference)


## Training

ორივე მოდელი გაწვრთნილია კონტრასტული სწავლების (contrastive learning) მეთოდით:

| Setting | Transformer (scratch) | DistilBERT |
|---|---|---|
| Loss | InfoNCE / Triplet | InfoNCE |
| Optimizer | AdamW | AdamW |
| Learning rate | 3e-4 | 2e-5 |
| Batch size | 64 | 32 |
| Epochs | 6 | 2 |

InfoNCE არის ძირითადი დანაკარგის ფუნქცია. ის ხელახლა იყენებს ბეჩში არსებულ ყველა სხვა პოზიტივს, როგორც ბეჩშიდა ნეგატივებს (in-batch negatives),
რაც უზრუნველყოფს $2B-1$ ნეგატივს თითოეულ ბიჯზე, Triplet loss-ის 1 ნეგატივის წინააღმდეგ,
და ყოველთვის იძლევა არანულოვან გრადიენტს. Triplet loss სატურაციას განიცდის როგორც კი ზღვარი (margin) დაკმაყოფილდება,
რის გამოც ჩვენს ექსპერიმენტებში მან საგრძნობლად დაბალი შედეგი აჩვენა.


## Results

მოდელები შეფასდა 500 სატესტო მოთხოვნაზე, 5,500 დოკუმენტისგან შემდგარ კორპუსში
(სატესტო პოზიტივებს + 5,000 საწვრთნელი გამაბნეველი/distractor პასაჟი).


| Model | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| TF-IDF (baseline) | 0.684 | 0.872 | 0.908 | 0.764 |
| Transformer-scratch (InfoNCE) | 0.400 | 0.550 | 0.622 | 0.464 |
| DistilBERT (InfoNCE) | 0.806 | 0.940 | 0.966 | 0.865 |

DistilBERT-ი ყველა მეტრიკაში სჯობს TF-IDF-ს. ნულიდან აგებული ტრანსფორმერი ჩამორჩება
TF-IDF-ს საწვრთნელი მონაცემების მცირე რაოდენობისა და შემთხვევითი ნეგატივების გამოყენების გამო — რთული ნეგატივების მოპოვება
(hard negative mining) ამ სხვაობას საგრძნობლად შეამცირებდა.


## Setup

```bash
pip install torch transformers scikit-learn datasets pypdf
```

**Prepare data:**
```bash
python src/data_prep.py
python src/fetch_book.py
```

**Train:**
```bash
python src/train.py --loss infonce --epochs 6 --model_type transformer_scratch
python src/train.py --loss infonce --epochs 2 --model_type transformer
```

**Evaluate:**
```bash
python src/evaluate.py --checkpoint checkpoints/encoder_tfscratch_infonce_final.pt
```

## Notebooks

| Notebook | Contents |
|---|---|
| `data_exploration.ipynb` | სიგრძეების განაწილება, მოთხოვნის ტიპები, ტოკენების გადაფარვის ანალიზი |
| `training.ipynb` | დანაკარგის (loss) მრუდები, ვალიდაციის მეტრიკები ეპოქების მიხედვით, მოდელების შედარება |
| `evaluation.ipynb` | საბოლოო სატესტო მეტრიკები, სვეტოვანი დიაგრამები (bar charts), წიგნში ძებნის თვისებრივი დემო |
