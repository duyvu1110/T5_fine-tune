import os
def convert_dataset(path):
    files = os.listdir(path)
    data_dict = {}
    for f in files:
        file_path = os.path.join(path, f)
        # a = read_file(file_path)
        # print(a)
        senandtuple = []

        with open(file_path, 'r', encoding='utf-8') as file:
            line = file.read().split('\n\n')
            for l in line:
                try:
                    sentence = l.split('\t')[1]
                except:
                    print(l)
                if 'alt' in l or 'des' in l or 'title' in l or 'Title' in l:
                    continue
                else:
                    senandtuple.append(sentence)
       # print(senandtuple[0])
convert_dataset('D:\T5_fine-tune\\test')