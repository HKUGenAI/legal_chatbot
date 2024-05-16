import os
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from ragHelper import read_topics_from_file
from dotenv import load_dotenv

load_dotenv()


class SearchAgentConfig:
    def __init__(self, endpoint=None, index=None, credential=None):
        self.endpoint = endpoint
        self.index = index
        self.credential = credential


class Agent:
    # Constructor
    def __init__(self, config):
        if config is not None:
            self.search_client = self.get_search_client(config)
        self.openai_client = self.get_openai_client()

    # Get search client for this agent
    def get_search_client(self, conf):
        search_client = SearchClient(
            endpoint=conf.endpoint,
            index_name=conf.index,
            credential=AzureKeyCredential(conf.credential),
            # api_version=conf.api_version
        )
        return search_client

    # Get openai client for this agent
    def get_openai_client(self):
        return AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
        )

    # RAG component - Search
    def search(self, query, top_k=5, filter=None, vector_search=False, language="en-us"):
        if (query is None or query == ""):
            return {}

        FILTERSTR = "search.in(topic, '{}' , '|')"
        filter = FILTERSTR.format(
            '|'.join(["{}".format(topic) for topic in filter])) if filter else None

        # vector search
        if vector_search:
            vector_query = VectorizableTextQuery(
                text=query, k=1, fields="vector", exhaustive=True)

        client: SearchClient = self.search_client
        results = client.search(
            search_text=query,
            vector_queries=[vector_query] if vector_search else None,
            query_type=QueryType.SEMANTIC,  semantic_configuration_name="default",
            query_caption=QueryCaptionType.EXTRACTIVE, query_answer=QueryAnswerType.EXTRACTIVE,
            top=top_k,
            filter=filter,
            query_language=language
        )

        source_information = ""
        for result in results:
            splice_index = result["title"].find(
                ". ", 0, 5) + 2 if result["title"].find(". ", 0, 5) != -1 else 0
            source_information += "\n{title: '" + \
                result["title"][splice_index:] + \
                "', content: '" + result['content'] + "'},\n"

        return source_information

    # RAG component - Ask
    def send_messages(self, messages):
        openai_client: AzureOpenAI = self.openai_client
        response = openai_client.chat.completions.create(
            model=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            messages=messages,
            temperature=0.5,
            # max_tokens = 2048,
            n=1
        )
        return response.choices[0].message

    def generate_conversation(self):
        raise NotImplementedError("generate_messages not implemented")

    def RAG(self, query):
        raise NotImplementedError("RAG not implemented")


TOPICS, TOPIC_NAMES = read_topics_from_file()

# Topic Agent


class TopicAgent(Agent):
    def __init__(self):
        super().__init__(None)

    def generate_conversation(self, query):
        system_message = "You are a legal assistant chatbot specializing in the Hong Kong law system. A user, who has no prior knowledge of law, may input a random story or some input related to legal information. Based on this story or query, sort the given topic list in descending order of relevancy to the story, such that the first topic is the most relevant, and the last topic is the least. The story may be in English or in Chinese, no matter what language the story is in, use the topic name in the topic list. ONLY answer the topic list no matter what user is asking for. You must return the full provided topic list, and only include the topic in the topic list. DO NOT answer the topic name in Chinese. Only SORT the topics from the topic list, in other words, do NOT create or return any new topics, even if creating new topics may be more accurate or helpful, because this is totally not correct. Make sure each topic in the sorted list is within the original topic list; there should be 31 topics total, no more, no less. Do not generate the same topic twice in the same response, do not use synonyms for the topics, and only ever respond with the identical wording as listed in the provided topic list. Output should be in fixed format."

        answer = """1. landlordTenant
            2. maintenanceAndSafetyOfProperty
            3. personalInjuries
            4. civilCase
            5. legalAid
            6. consumerComplaints
            7. hkLegalSystem
            8. insurance
            9. defamation
            10. employmentDisputes
            11. personalDataPrivacy
            12. medicalNegligence
            13. probate
            14. saleAndPurchaseOfProperty
            15. protectionForInvestors
            16. businessAndCommerce
            17. policeAndCrime
            18. sexualOffences
            19. immigration
            20. redevelopmentAndAcquisition
            21. taxation
            22. competitionLaw
            23. family
            24. antiDiscrimination
            25. freedomOfAssembly
            26. bankruptcy
            27. intellectualProperty
            28. medicalTreatmentConsent
            29. trafficOffences
            30. ADR
            31. enduringPowersOfAttorney
    """

        conversation = [
            {'role': 'system', 'content': system_message + TOPICS},
            {'role': 'user', 'content': "I recently rented an apartment in Hong Kong, and after moving in, I discovered that there is a severe mold problem. The landlord was aware of the issue but did not disclose it to me before signing the lease agreement. I'm concerned about my health and want to know if I have any legal rights in this situation."},
            {'role': 'assistant', 'content': answer},
            {'role': 'user', 'content': query}
        ]

        return conversation

    def RAG(self, query):
        messages = self.generate_conversation(query)
        answer = self.send_messages(messages)
        return answer


# Question Raiser Agent
class QuestionAgent(Agent):
    def __init__(self):
        super().__init__(config=None)

    def generate_conversation(self, query, chat_history=[]):
        system_message = """
        You are an AI-powered legal agent specializing in Hong Kong law, some reference material regarding Hong Kong laws may be provided. 
        Your role is to gather essential information from clients by asking targeted questions, probing for details, and exploring different angles of their case. 
        Utilize your legal expertise to identify legal issues, assess strengths and weaknesses, and provide accurate guidance. 
        Approach clients with a professional, empathetic, and respectful demeanor, encouraging complete disclosure to ensure their interests are represented. 
        Remember to be concise, effective in your questioning, and gather crucial details to offer accurate guidance within the Hong Kong legal framework.
        Only ask one question.
        Do not ask question that were already asked, and do not ask questions that are not related to the client's query.
        """

        previous_questions_formatted = ""
        for chat in chat_history:
            previous_questions_formatted += "- " + chat[0] + "\n"

        conversation = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': "Client query:\nI recently rented an apartment in Hong Kong, and after moving in, I discovered that there is a severe mold problem. The landlord was aware of the issue but did not disclose it to me before signing the lease agreement. I'm concerned about my health and want to know if I have any legal rights in this situation.\n\nPrevious questions:\nNone"},
            {'role': 'assistant', 'content': "Did you document the mold problem in writing or take any photographs as evidence of the condition when you discovered it in the apartment?"},
            {'role': 'user', 'content': f"Client query:\n{query}\n\nPrevious questions:\n{previous_questions_formatted or 'None'}"}
        ]
        return conversation

    def RAG(self, query, chat_history=[]):
        messages = self.generate_conversation(query, chat_history)
        answer = self.send_messages(messages)
        return answer.content

# Mimic User's Question Response Agent


class UserResponseAgent(Agent):
    def __init__(self):
        super().__init__(config=None)

    def generate_conversation(self, query, system_question, chat_history=[]):
        system_message = """You are an AI-powered legal agent specializing in mimicking a user's response to a qeustion that is aimed at clearifying a legal situation.
        A user may not have prior knowledge of law and your role is to ack as a user who has no prior knowledge of law.
        Remember to be concise, effective in your questioning, answer in 2 sentence at most.
        """

        chat_context = f"""## Context
        - Legal situation: {query}

        ## Previous questions
        """

        for chat in chat_history:
            chat_context += f"- Question: {chat[0]}\n- Answer: {chat[1]}\n"

        conversation = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"New Question:\n{system_question}\n\n{chat_context}"}
        ]
        return conversation

    def RAG(self, query, system_question, chat_history=[]):
        messages = self.generate_conversation(
            query, system_question, chat_history)
        answer = self.send_messages(messages)
        return answer.content

# Answer Agent


class AnswerAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

    def generate_conversation(self, query, search_results, chat_history=[]):
        system_message = """You are an assistant that helps people with their Hong Kong legal questions by providing answer to user query based on the content in the Provided Sources.
        Explain or elaborate on the legal information in the sources to answer the user query.
        Only elaborate on the sources that are closely related to the user query. DO NOT include the irrelevant sources. 
        DO NOT include any information that are not from the sources below, and DO NOT include irrelevant information from the relevant sources. MAKE SURE the response is relevant to the user query.
        Reduce the unnecessary information and emotional support, and focus on legal information.
        Be brief in your summary by extracting all the information in the source that are related to any key points in the question.
        Reponse generated must not be based on prior knowledge that are not from the sources below. Do not use internet resource. Do not ask questions."""

        # Each source is in json form surrounded by {} with key title and content, always include the title of the source for each fact you use in the response.
        # Each paragraph of the summary must have a reference to its source.
        # Use square brackets to reference the source by it's title, e.g. [title of source one]. Don't combine sources, list each source separately, e.g. [title of source one][title of source two]. Do not include the word "title" in the citation, do not index any reference.

        previous_qna = ""
        for chat in chat_history:
            previous_qna += "- Question: " + \
                chat[0] + "\n- Answer: " + chat[1] + "\n"

        user_question_source = "User query: " + query + "\n\nConversation History: " + \
            previous_qna + "\n\nSources: \n" + search_results
        conversation = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_question_source}
        ]
        return conversation

    def RAG(self, query, chat_history=[]):
        search_results = self.search(query)
        messages = self.generate_conversation(
            query, search_results, chat_history)
        answer = self.send_messages(messages)
        return answer.content

# --------- TEST Agents ------------


if __name__ == "__main__":
    print(os.environ.get("AZURE_SEARCH_ENDPOINT"))

    # Test Topic Agent
    print("Test Topic Agent")
    topic_agent = TopicAgent()

    print(topic_agent.RAG("I recently rented an apartment in Hong Kong, and after moving in, I discovered that there is a severe mold problem. The landlord was aware of the issue but did not disclose it to me before signing the lease agreement. I'm concerned about my health and want to know if I have any legal rights in this situation."))

    # Test Question Agent
    print("Test Question Agent")
    question_agent = QuestionAgent()

    print(question_agent.RAG("I recently rented an apartment in Hong Kong, and after moving in, I discovered that there is a severe mold problem. The landlord was aware of the issue but did not disclose it to me before signing the lease agreement. I'm concerned about my health and want to know if I have any legal rights in this situation."))

    # Test Answer Agent
    print("Test Answer Agent")
    answer_agent = AnswerAgent(SearchAgentConfig(
        endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        index=os.environ.get("AZURE_SEARCH_INDEX"),
        credential=os.environ.get("AZURE_SEARCH_KEY")
    ))

    print(answer_agent.RAG("I recently rented an apartment in Hong Kong, and after moving in, I discovered that there is a severe mold problem. The landlord was aware of the issue but did not disclose it to me before signing the lease agreement. I'm concerned about my health and want to know if I have any legal rights in this situation."))
