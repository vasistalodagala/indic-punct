import pandas as pd
import csv
import re

from inverse_text_normalization.run_predict import inverse_normalize_text
x = inverse_normalize_text(['शून्य'],'mr')
print(x)

def load_data():
  input = []
  output = []

  train_tsv_file = open('/IITM/Sem3/MTP/data/num_words_mr.tsv', "r", encoding="utf-8")
  train_dataset = csv.reader(train_tsv_file, delimiter="\t")
  for i in train_dataset:
    input.append(i[0])
    output.append(i[1])

  return input, output

def preprocess(input):
  '''
    Removing extra , in the input
  '''
  for i in range (len(input)):
    input[i] = input[i].replace(' ,', '')
  return input


input, output = load_data()
input_new = preprocess(input)
input_new[:5]

count = 0
f = open('/IITM/Sem3/MTP/data/number_words_mr_mismatch.tsv', 'w') 
line = 'ASR output' + '\t' + 'model output (predicted)' + '\t' + 'expected output' + '\n'
f.write(line)

for i in range(500):
    l = []
    pred = inverse_normalize_text([input_new[i]],'mr')
    # pred = int(pred[0])
    l.append(str(output[i]))
    if(pred != l ):
        print(pred, l)
        count+=1
        #writing to the file
        line = input_new[i] + '\t' + pred[0] + '\t' + l[0] + '\n'
        f.write(line)
print(count)
f.close()