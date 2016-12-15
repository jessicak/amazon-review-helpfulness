num_reviews = 222240 # Number of reviews per category
final_out_fn = 'deskewed_dataset.json'

import random

entries = []
new_line_char = '\n'

for i in range(5):
    dataset_i = 'dataset' + str(i) + '_sorted.json'
    num_recorded = 0
    with open(dataset_i) as f:
        for line in f:
            if line[-1] == new_line_char:
                line = line[:-1]
            entries.append(line)
            num_recorded += 1
            if num_recorded == num_reviews:
                break

random.shuffle(entries)

out = open(final_out_fn, 'a')
for entry in entries:
    out.write(entry + new_line_char)
out.close()
