from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
model = AutoModelForSeq2SeqLM.from_pretrained('duyvu8373/viT5-base-coqe')
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
