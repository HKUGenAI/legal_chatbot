import csv
    
def read_topics_from_file():
    TOPIC_NAMES = {}
    TOPICS = ""
    with open("./topic_translation.csv", 'r', encoding='utf8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row if present
        for row in csv_reader:
            key = row[0]
            TOPICS += key + ", "
            en = row[1]
            tc = row[2]
            sc = row[3]
            TOPIC_NAMES[key] = {'en-US': en, 'zh-HK': tc, 'zh-CN': sc}

    TOPICS = 'Topic List: (' + TOPICS[:-2] + ')'
    return TOPICS, TOPIC_NAMES
    
# TOPICS, TOPIC_NAMES = read_topics_from_file()
# print(TOPICS)
# print(TOPIC_NAMES)