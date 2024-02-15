import dotenv
import evaluate
import numpy as np
from openai import AzureOpenAI

dotenv.load_dotenv()

class EvalAgent:
    def __init__(self, method = "opanAIEmbedding"):
        self._method = method

    def evaluvate(self, prediction, reference):
        if self._method == "rouge":
            return self.evalROUGE(prediction, reference)
        elif self._method == "bleu":
            return self.evalBLEU(prediction, reference)
        elif self._method == "BERTScore":
            return self.evalBERTScore(prediction, reference)
        elif self._method == "OpenAIEmbedding":
            return self.evalOpenAIEmbedding(prediction, reference)
    
    def evalROUGE(self, prediction, reference):
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=[prediction], references=[reference])
        return results
    
    def evalBLEU(self, prediction, reference):
        bleu = evaluate.load('bleu')
        results = bleu.compute(predictions=[prediction], references=[reference])
        return results
    
    def evalBERTScore(self, prediction, reference):
        bert = evaluate.load('bertscore')
        results = bert.compute(predictions=[prediction], references=[reference], lang='en')
        return results
    
    def evalOpenAIEmbedding(self, prediction, references):
        client = AzureOpenAI(
            api_version = "2024-02-15-preview",
        )
        prediction_embedding = client.embeddings.create(
            model = "embedding",
            input = prediction,
        ).data[0].embedding
        references_embedding = client.embeddings.create(
            model = "embedding",
            input = references,
        ).data[0].embedding
        return np.dot(prediction_embedding, references_embedding)



if __name__ == "__main__":
    # print(EvalAgent('BERTScore').evaluvate("hello here", "how are you"))
    print(EvalAgent('OpenAIEmbedding').evaluvate("hello here", "how are you"))