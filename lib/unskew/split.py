"""
Split dataset by helpfulness review
"""

dataset_fn0 = 'dataset0.json'
dataset_fn1 = 'dataset1.json'
dataset_fn2 = 'dataset2.json'
dataset_fn3 = 'dataset3.json'
dataset_fn4 = 'dataset4.json'

json_fn_list = '5core_json_list.txt'
min_feedback = 5

buffers = {
    0 : [],
    1 : [],
    2 : [],
    3 : [],
    4 : []
}

outs = {}

max_buf_len = 1000

try:
    import ujson as json
except:
    import json

def get_json_fns():
    json_fn = []
    with open(json_fn_list) as f:
        for line in f:
            if len(line) > 2 and line[0] != '#':
                json_fn.append(line.strip())
    return json_fn

def process_datafile(fn, metrics):
    print "Processing " + fn
    category_name = fn[8:-7]
    with open(fn) as f:
        for line in f:
            if len(line) < 10:
                continue
            d = json.loads(line)
            text = d['reviewText']
            h1, h2 = d['helpful']
            if len(text) < 5:
                continue
            category_id = log_metrics(metrics, h1, h2)
            if category_id >= 0:
                buf = buffers[category_id]
                # Modify JSON object
                d.pop('reviewTime')
                d.pop('unixReviewTime')
                if 'reviewerName' in d:
                    d.pop('reviewerName')
                d['category'] = category_name
                buf.append(json.dumps(d))
                if len(buf) >= max_buf_len:
                    out_file = outs[category_id]
                    for entry in buf:
                        out_file.write(entry + '\n')
                    buffers[category_id] = []
    
    for k in buffers:
        buf = buffers[k]
        out_file = outs[k]
        for entry in buf:
            out_file.write(entry + '\n')
        buffers[k] = []

def log_metrics(metrics, n, d):
    if len(metrics) == 0: # Set up metrics dictionary
        metrics[0] = 0
        metrics[1] = 0
        metrics[2] = 0
        metrics[3] = 0
        metrics[4] = 0
    if n > d:
        return -1
    if d >= min_feedback:
        s = (1.0 * n) / (1.0 * d)
        if s <= 0.20:
            cat = 0
        elif s <= 0.40:
            cat = 1
        elif s <= 0.60:
            cat = 2
        elif s <= 0.80:
            cat = 3
        elif s <= 1.00:
            cat = 4
        metrics[cat] += 1
        return cat
    return -1

def main(metrics={}):
    json_fns = get_json_fns()
    out_0 = open(dataset_fn0,'a')
    out_1 = open(dataset_fn1,'a')
    out_2 = open(dataset_fn2,'a')
    out_3 = open(dataset_fn3,'a')
    out_4 = open(dataset_fn4,'a')


    global outs 
    outs = {
        0: out_0,
        1: out_1,
        2: out_2,
        3: out_3,
        4: out_4
    }

    for json_fn in json_fns:
        process_datafile(json_fn, metrics)
    out_0.close()
    out_1.close()
    out_2.close()
    out_3.close()
    out_4.close()
    outs = {}
    print metrics

if __name__ == "__main__":
    metrics = {}
    main(metrics=metrics)
