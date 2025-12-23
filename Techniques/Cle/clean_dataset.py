import json
import argparse
import glob
import pprint

parser = argparse.ArgumentParser(description='Clean dataset.')
parser.add_argument('json_filename', type=str, help='Filename of JSON containing dataset')

args = parser.parse_args()

dataset_json = args.json_filename

defective_lines_filename = glob.glob(f'defective_lines_*.json')[0]
syntax_errors_filename = glob.glob(f'syntax_errors_*.json')[0]

with open(dataset_json, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
print("Training set loaded")

with open(defective_lines_filename, 'r', encoding='utf-8') as def_f:
    defective_lines = json.load(def_f)
print("Defective lines loaded")

with open(syntax_errors_filename, 'r', encoding='utf-8') as syn_f:
    syntax_errors_lines = json.load(syn_f)
print("Syntax errors lines loaded")  

print(len(defective_lines))
print(len(syntax_errors_lines))
print(list(set(defective_lines) & set(syntax_errors_lines)))

syntax_iter = iter(syntax_errors_lines)
defective_iter = iter(defective_lines)

current_syntax_line = next(syntax_iter, None)
current_defective_line = next(defective_iter, None)

syntax_errors_samples = []
defective_samples = []
clean_samples = []

for line, sample in enumerate(dataset):
    if line != 0 and line % 100000 == 0:
        print(f'Processed {line} lines...')
    
    if current_syntax_line is not None and line == current_syntax_line:
        syntax_errors_samples.append(sample)
        current_syntax_line = next(syntax_iter, None)  
    elif current_defective_line is not None and line == current_defective_line:
        defective_samples.append(sample)
        current_defective_line = next(defective_iter, None) 
    else:
        clean_samples.append(sample)

with open('defective_samples_'+dataset_json, 'w', encoding='utf-8') as f:
        json.dump((defective_samples), f)

with open('syntax_errors_samples_'+dataset_json, 'w', encoding='utf-8') as f:
        json.dump((syntax_errors_samples), f)

with open('clean_samples_'+dataset_json, 'w', encoding='utf-8') as f:
        json.dump((clean_samples), f)


print(f"Extracted {len(defective_samples)} defective samples from the dataset.")
print(f"Extracted {len(syntax_errors_samples)} samples with syntax errors from the dataset.")
print(f"Original dataset size: {len(dataset)}. Cleaned dataset size: {len(clean_samples)}.")
