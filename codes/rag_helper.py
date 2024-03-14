
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType
from azure.search.documents import SearchClient

    
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