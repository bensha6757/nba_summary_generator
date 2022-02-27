import re


def remove_blacklist(blacklist, summary):
    clean_summary = []
    i = 0
    sentences = summary.split(".")
    n = len(sentences)
    for sentence in sentences:
        if i == 0 or i == 1 or not is_black_list_in_sent(sentence, blacklist):
            if i == n - 1 and sentence == '\n':
                clean_summary.pop()
                clean_summary.append(sentence)
            elif i != n - 1:
                clean_summary.append(sentence)
            else:
                clean_summary.append('\n')
        else:
            clean_sent = remove_sub_sent(sentence, blacklist)
            if clean_sent:
                clean_summary.append(remove_sub_sent(sentence, blacklist))
        i += 1
    return '.'.join(clean_summary)


def is_black_list_in_sent(sent, blacklist):
    return re.search(blacklist, sent, flags=re.IGNORECASE)


def remove_sub_sent(sent, blacklist):
    clean_sent = []
    for sub_sent in sent.split(","):
        if not is_black_list_in_sent(sub_sent, blacklist):
            clean_sent.append(sub_sent)
    return ','.join(clean_sent)


def read_summaries(filename, outfilename, blacklist):
    with open(filename, "r", encoding="utf8") as input_file:
        with open(outfilename, "w") as output:
            for line in input_file:
                output.write(remove_blacklist(blacklist, line))


if __name__ == '__main__':
    blacklist = 'injured|injury|injuries|referee|refereeing|referees|judge|judges|judging|judgment|all-star|allstar' \
                '|all star|knee|knees|shoulder|shoulders|series|serieses|Achilles|neck|hamstring|Next ,' \
                '|\(.[a-z]+.\)|playoff|playoffs|will head|next game|next games|will travel|road trip|road ' \
                'trips|road-trip '
    input_filename = "C:/Users/ben.shapira/OneDrive - Algosec Systems " \
                     "Ltd/Desktop/University/NLP/Project/Data/exported_files/D1_2014_text.txt"
    outfilename = "C:/Users/ben.shapira/OneDrive - Algosec Systems " \
                  "Ltd/Desktop/University/NLP/Project/Data/exported_files/D1_2014_text_clean.txt"
    read_summaries(input_filename, outfilename, blacklist)
