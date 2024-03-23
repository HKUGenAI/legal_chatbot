from agents import QuestionAgent, AnswerAgent, UserResponseAgent, SearchAgentConfig
from evalAgent import EvalAgent
import os, logging

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.8
SEARCH_CONFIG = SearchAgentConfig(
        endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        index=os.environ.get("AZURE_SEARCH_INDEX"),
        credential=os.environ.get("AZURE_SEARCH_KEY")
    )

class chat:
    def __init__(self):
        self._questionAgent = QuestionAgent()
        self._answerAgent = AnswerAgent(SEARCH_CONFIG)
        self._userResponseAgent = UserResponseAgent()
        self._evalAgent = EvalAgent('OpenAIEmbedding')
        self._chatHistory = []
        self._userQuery = ""

    def appendToChatHistory(self, sysMsg, userMsg):
        self._chatHistory.append((sysMsg, userMsg))

    def run(self):
        query = input("User: ")
        self._userQuery = query

        question = self._questionAgent.RAG(query)
        print(f"Sys: {question}")
        answer = input("User: ")
        self._chatHistory.append((question, answer))

        while True:
            if answer == "exit":
                break

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
                exit()
            else:
                print(f"Sys: {question}")
                answer = input("User: ")
                self._chatHistory.append((question, answer))

if __name__ == "__main__":
    logging.basicConfig(filename='run.log', level=logging.INFO)
    new_chat = chat()
    new_chat.run()