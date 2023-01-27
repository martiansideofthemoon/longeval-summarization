import csv
import json
import pickle
import string
import spacy
import subprocess

try:
    nlp = spacy.load('en_core_web_lg')
    # nlp.add_pipe('sentencizer', before="parser")
except OSError:
    raise OSError("'en_core_web_lg model' is required unless loading from cached file."
                    "To install: 'python -m spacy download en_core_web_lg'")


def split_sents(text, length_threshold=5):
    """Break up sentences into clauses and phrases."""
    sents = get_sents(text, return_nlp=True)
    all_sent_units = []
    for s1 in sents:
        if not s1.text.strip():
            continue
        pos_tags = [x.pos_ for x in s1]
        tokens = [x.text for x in s1]
        all_units = []
        for i, (pt, tk) in enumerate(zip(pos_tags, tokens)):
            if tk in [',', '!', '?', ';'] or 'CONJ' in pt:
                if tk == "that":
                    continue
                all_units.append(i)

        all_units.append(len(s1))


        # combine short units together
        combined_units = []
        for i in range(0, len(all_units)):
            if i == 0 or (all_units[i] - all_units[i - 1]) > length_threshold:
                combined_units.append(all_units[i])
            else:
                combined_units[-1] = all_units[i]

        if combined_units[0] <= length_threshold:
            combined_units = combined_units[1:]

        unit_strs = []
        for i in range(len(combined_units)):
            if i == 0:
                unit_strs.append(s1[0:combined_units[i]].text)
            else:
                unit_strs.append(s1[combined_units[i - 1] + 1:combined_units[i]].text)
        all_sent_units.append(unit_strs)

    return all_sent_units, [x.text for x in sents]


def pickle_write(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def pickle_read(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def csv_write(data_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for x in data_list:
            spamwriter.writerow(x)

def jsonl_read(filename):
    dataset = []
    with open(filename, 'r') as f:
        dataset = [json.loads(x) for x in f.read().strip().split("\n")]
    return dataset

def jsonl_write(filename, dataset):
    with open(filename, 'w') as f:
        f.write("\n".join([json.dumps(x) for x in dataset]) + "\n")

def json_read(filename):
    dataset = []
    with open(filename, 'r') as f:
        dataset = json.loads(f.read())
    return dataset

def csv_read(filename, header=True):
    dataset = []
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            dataset.append(row)
    if header:
        return dataset[0], dataset[1:]
    else:
        return None, dataset

def csv_dict_read(filename):
    dataset = []
    with open(filename, 'r') as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            dataset.append(row)
    return dataset

def csv_dict_write(dataset, filename):
    fields = dataset[0].keys()
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fields)
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)
    return dataset

def detokenize(text):
    text = text.replace(" .", ".")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = text.replace(" ,", ",")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    text = text.replace(" %", "%")
    text = text.replace(" )", ")")
    return text

def process_ms2(text):
    text = text.split()
    # remove tags
    text = filter(lambda x: not x.startswith("<") or not x.endswith(">"), text)
    text = detokenize(" ".join(text))
    if not text.endswith(string.punctuation):
        text = text + "."
    return text

def get_sents(text, return_nlp=False):
    if not isinstance(text, str):
        # assume it is list or tuple of strings
        return text
    global nlp
    nlp_sents = [x for x in nlp(text).sents]
    if return_nlp:
        return nlp_sents
    else:
        return [x.text for x in nlp_sents]

class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def postprocess(cls, input_str):
        input_str = input_str.replace("<h>", cls.HEADER)
        input_str = input_str.replace("<blue>", cls.OKBLUE)
        input_str = input_str.replace("<green>", cls.OKGREEN)
        input_str = input_str.replace("<yellow>", cls.WARNING)
        input_str = input_str.replace("<red>", cls.FAIL)
        input_str = input_str.replace("</>", cls.ENDC)
        input_str = input_str.replace("<b>", cls.BOLD)
        input_str = input_str.replace("<u>", cls.UNDERLINE)
        input_str = input_str.replace("<clean>", "")
        return input_str


def export_server(output, filename):
    with open("{}.txt".format(filename), "w") as f:
        f.write(Bcolors.postprocess(output) + "\n")
    subprocess.check_output("cat {0}.txt | ansi2html.sh --palette=linux --bg=dark > {0}.html".format(filename), shell=True)
    subprocess.check_output("rm {}.txt".format(filename), shell=True)

