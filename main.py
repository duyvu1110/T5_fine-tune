import  os
import re
import json
from datasets import Dataset, load_from_disk
from datasets import load_dataset
import pyarrow as pa
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


pattern = ('\{.*"subject": \[.*\], "object": \[.*\], "aspect": \[.*\], "predicate": \[.*\], "label": ".*"\}')
def read_file(f_name):
    data = []
    with open(f_name, 'r',encoding = 'utf-8') as f:
        sent_tuples = []
        txt = False
        for l in f:
            l = l.strip()

            if len(l) == 0:
                if txt:
                    data.append(sent_tuples)
                sent_tuples = []
                txt = False
            elif l.startswith('{'):
                try :
                    json.loads(l)
                except:
                    print(l)
                    print(f)
                sent_tuples.append(json.loads(l))
            else:
                # text line
                txt = True

        if txt:
            data.append(sent_tuples)

    return data
def convert_quintuple(q):
    quintuple = json.loads(q)
    subject_value = " ".join([s.split("&&")[1] for s in quintuple["subject"]])
    object_value = " ".join([o.split("&&")[1] for o in quintuple["object"]])
    aspect_value = " ".join([a.split("&&")[1] for a in quintuple["aspect"]])
    predicate_value = " ".join([p.split("&&")[1] for p in quintuple["predicate"]])
    label_value = quintuple["label"]
    if len(quintuple["subject"]) == 0:
        subject_value = 'None'
    if len(quintuple["object"]) == 0:
        object_value = 'None'
    if len(quintuple["aspect"]) == 0:
        aspect_value = 'None'
    formatted_output = f"{subject_value}; {object_value}; {aspect_value}; {predicate_value}; {label_value}"
    res = '(' + formatted_output + ')'
    return res
def convert_dataset(path):
    files = os.listdir(path)
    data_dict = {}
    for f in files:
        file_path = os.path.join(path,f)
        # a = read_file(file_path)
        # print(a)
        senandtuple = []

        with open(file_path, 'r', encoding= 'utf-8') as file:
            line = file.read().split('\n\n')
            for l in line:
                if l == '':
                    continue
                sentence = l.split('\t')[1]
                if 'alt' in l or 'des' in l or 'title' in l or 'Title' in l:
                    continue
                else:
                    senandtuple.append(sentence)
        quintuple = []
        current_sentence = None
        for item in senandtuple:
            a = item.split('\n')
            if len(a) == 1:
                data_dict[a[0]] = "(None;None;None;None;None)"
            elif len(a) == 2:
                data_dict[a[0]] = convert_quintuple(a[1])
            elif len(a) > 2:
                list_quintuple = ''
                for i in range(1,len(a)):
                    if i != len(a) -1:
                        list_quintuple = list_quintuple + convert_quintuple(a[i]) + ';'
                    else:
                        list_quintuple = list_quintuple + convert_quintuple(a[i])
                data_dict[a[0]] = list_quintuple
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    hg_ds = Dataset.hg_dataset = Dataset(pa.Table.from_pandas(df))
    return hg_ds
if __name__ == '__main__':
    # train_ds = convert_dataset('D:\T5_fine-tune\VLSP2023_ComOM_training_v2')
    # train_ds.save_to_disk('train_dataset')
    # dev_ds = convert_dataset('D:\T5_fine-tune\VLSP2023_ComOM_dev_v2')
    # dev_ds.save_to_disk('dev_dataset')
    # test_ds = convert_dataset('D:\T5_fine-tune\VLSP2023_ComOM_testing_v2')
    # test_ds.save_to_disk('test_dataset')
    train_ds = load_from_disk('train_dataset')
    dev_ds  = load_from_disk('dev_dataset')
    test_ds = load_from_disk('test_dataset')
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
    model.cuda()
    prefix = 'Please extract five elements including subject, object, aspect, predicate, and comparison type in the sentence'
    max_input_length = 156
    max_target_length = 156
    print(train_ds[0])
    print(train_ds[0][0])

    # def preprocess_function(examples):
    #     inputs = prefix
    #     for ex in examples:
    #         inputs = inputs + ex.split('__index_level_0__')[1]
    #     targets = [ex.split('0')[1].split('__index_level_0__')[0] for ex in examples]
    #     model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    #     # Setup the tokenizer for targets
    #     with tokenizer.as_target_tokenizer():
    #         labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs
    # tokenized_ds = train_ds.map(preprocess_function, batched = True)
    # print(tokenized_ds)