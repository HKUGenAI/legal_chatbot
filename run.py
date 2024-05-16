from agents import QuestionAgent, AnswerAgent, UserResponseAgent, SearchAgentConfig
from evalAgent import EvalAgent
import multiprocessing as mp
import os
import logging

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.85
SEARCH_CONFIG = SearchAgentConfig(
        endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        index=os.environ.get("AZURE_SEARCH_INDEX"),
        credential=os.environ.get("AZURE_SEARCH_KEY")
    )
answerAgent = AnswerAgent(SEARCH_CONFIG)

def generateResponseMP(query, chatHistory, responseRef) -> None:
    response = answerAgent.RAG(query, chatHistory)
    responseRef.value = response

class Chat:
    def __init__(self):
        self._questionAgent = QuestionAgent()
        self._answerAgent = answerAgent
        self._userResponseAgent = UserResponseAgent()
        self._evalAgent = EvalAgent('OpenAIEmbedding')
        self._chatHistory = []
        self._userQuery = ""

    def appendToChatHistory(self, sysMsg, userMsg):
        self._chatHistory.append((sysMsg, userMsg))

    def complete(self, query):
        if self._userQuery == "":
            self._userQuery = query
            question = self._questionAgent.RAG(query)
            self._previous_question = question
            return question, None, False
        
        self._chatHistory.append((self._previous_question, query))
        # Generate new question
        question = self._questionAgent.RAG(query, self._chatHistory)
        # Generate mock answer
        mock_answer = self._userResponseAgent.RAG(
            query, question, self._chatHistory)
        # Gerenate dummy response
        dummy_response = self._answerAgent.RAG(query, self._chatHistory + [(question, mock_answer)])

        # Generate real query response for the current round
        response = self._answerAgent.RAG(query, self._chatHistory)

        # Bepare similarity
        similarity = self._evalAgent.evaluvate(response, dummy_response)
        if (similarity >= SIMILARITY_THRESHOLD):
            # Generate query response
            self._previous_question = response
            return response, (question, mock_answer, response, dummy_response, similarity), False
        else:
            self._previous_question = question
            return question, (question, mock_answer, response, dummy_response, similarity), False


    def run(self):
        # query = input("User: ")
        # self._userQuery = query

        # question = self._questionAgent.RAG(query)
        # print(f"Sys: {question}")
        # answer = input("User: ")
        # self._chatHistory.append((question, answer))
        query = input("User: ")
        answer = ""
        while True:
            if answer == "exit()":
                break
            responseRef = mp.Manager().Value(str, "")
            responseMP = mp.Process(target=generateResponseMP, args=(query, self._chatHistory, responseRef))
            responseMP.start()

            # Generate new question
            question = self._questionAgent.RAG(query, self._chatHistory)
            # Generate mock answer
            mock_answer = self._userResponseAgent.RAG(
                query, question, self._chatHistory)
            # Gerenate dummy response
            dummy_response = self._answerAgent.RAG(query, self._chatHistory + [(question, mock_answer)])

            # Generate real query response for the current round
            # response = self._answerAgent.RAG(query, self._chatHistory)
            responseMP.join()
            response = responseRef.value

            # Bepare similarity
            similarity = self._evalAgent.evaluvate(response, dummy_response)
            print(similarity)

            #log
            logger.info("##################################")
            logger.info(f"next question: {question}")
            logger.info(f"mock answer: {mock_answer}")
            logger.info(f"dummy response: {dummy_response}")
            logger.info(f"current round response: {response}")
            logger.info(f"similarity: {similarity}")
            logger.info("##################################")

            if (similarity >= SIMILARITY_THRESHOLD):
                # Generate query response
                print(f"Sys: {response}")
                answer = input("User: ")
                self._chatHistory.append((response, answer))
                #exit()
            else:
                print(f"Sys: {question}")
                answer = input("User: ")
                self._chatHistory.append((question, answer))

if __name__ == "__main__":
    logging.basicConfig(filename='run.log', level=logging.INFO)
    new_chat = Chat()
    new_chat.run()