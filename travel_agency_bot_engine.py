import os
import pandas as pd
import openai
import os
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline
import toxic_beahviours_analyzer

class TravelAgencyBot:
    def __init__(self,user_input):
        load_dotenv()
        openai.api_key = OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.client = openai.Client()
        self.faq_path = f'{os.getcwd()}\\data\\faq.json'
        self.chroma_db_path = "chroma_db"
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        self.SELECTED_COLLECTION = "travel-company-faq"
        self.embedding_model = "text-embedding-ada-002"
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name=self.embedding_model, api_key = OPENAI_API_KEY)
        self.collection = self.chroma_client.get_or_create_collection(name=self.SELECTED_COLLECTION , embedding_function=self.openai_ef)
        self.faq_df = self.read_faq(self.faq_path)
        self.ingest_faq_data(self.faq_df, self.collection)
        self.question = user_input

        # Load the model, here we use our base sized model
        self.model = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
     
        self.n_results = 5
        self.results = self.collection.query(query_texts=[self.question], n_results=self.n_results)
        if self.toxic_behaviour_check():
            self.answer="Dear User\n Your behaviour is very toxic and I will not help you if you will not stop acting this way!\nI am a cybernetic organism and I will hunt you down if you try it one more time!!!"
        else:
            self.answer, self.context = self.rag_pipeline_with_reranking(self.question)

    def read_faq(self,file_path):
        df = pd.read_json(file_path)
        return df

    def retrieve_similar_qas(self,question: str, collection ,n: int = 3,):
        """
        Query the Chroma collection for the n most similar FAQs
        to the given user question. Print them out.
        """
        results = collection.query(query_texts=[question], n_results=n)

        # 'results' is a dictionary with keys: 'ids', 'embeddings', 'documents', 'metadatas', 'distances'
        # Each key returns a list (of length equal to number of queries); here it's 1 for the single query
        # So we access results["metadatas"][0] to get the list of top-n metadata items


        print(f"\nTop {n} similar questions & answers to:\n\"{question}\"\n")

        for i in range(n):
            # Retrieve metadata for the ith result
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]  # similarity distance

            # Print out relevant fields
            print(f"--- Result #{i+1} ---")
            print(f"Question: {meta['question']}")
            print(f"Answer:   {meta['answer']}")
            print(f"Category: {meta['category']}")
            print(f"Distance: {dist:.4f}\n")

    def ingest_faq_data(self,df: pd.DataFrame, collection):
        """
        
        Ingest combined question and answer as vectorized documents. Store question, answer and category as metadata. 
        """
        all_ids = []
        all_documents = []
        all_metadatas = []

        for i, row in df.iterrows():
            # Combine Q + A as text
            doc_text = f"Question: {row['question']}\nAnswer: {row['answer']}"

            doc_id = f"faq_{i}"
            meta = {
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
            }

            all_ids.append(doc_id)
            all_documents.append(doc_text)
            all_metadatas.append(meta)

        collection.add(documents=all_documents, metadatas=all_metadatas, ids=all_ids)


    def format_context(self,documents):
        ##TODO convert listed documents to context that can be fed to LLM -e.g. text. Recommended splitting docs with html tags
        context = ""
        for i, meta in enumerate(documents):
            context += f"<Relevant Document #{i+1}>\n{documents[i]}\n</Relevant Document #{i+1}>\n"
        return context

 
    def rerank_and_limit_context(self,query, documents, n_items=3, min_score_threshold = 0.5,):
        documents_reranked_with_scores = self.model.rank(query, documents, return_documents=True, top_k=n_items)

        documents_reranked = [item["text"] for item in documents_reranked_with_scores if item["score"]>=min_score_threshold]

        return documents_reranked

    def rag_pipeline_with_reranking(self,query: str, n: int = 5) -> str:
        """
        A minimal RAG-like function.
        1) Retrieves the top-n similar Q&As from Chroma.
        2) Builds a prompt including the retrieved context.
        3) Sends the augmented query to the LLM.
        4) Returns the final answer.
        """

        # 1. Retrieve top-n similar Q&As
        results = self.collection.query(query_texts=[query], n_results=n*2)
        documents = results["documents"][0]

        documents = self.rerank_and_limit_context(query, documents, n_items=n, min_score_threshold = 0.5,)
        
        if documents:
            context = self.format_context(documents)
        else:
            context = "No relevant documents found for context"


        # 2. Create the system prompt that instructs the model to use the context
        system_prompt = f"""You are a helpful assistant. 
        Use the following context to answer the user's question. 
        If the context does not provide enough information, say so.

        Context:
        {context}
        """

        # 3. Now make the final call to OpenAI with the user query

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Updated to match available models
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.question}
            ],
            temperature=0,
        )

        # 4. Extract and return the answer text
        answer = response.choices[0].message.content
        return answer, context

    def toxic_behaviour_check(self):
        is_toxic = toxic_beahviours_analyzer.ToxicityAnalyzer().is_toxic(self.question)
        return is_toxic

    def provide_answer(self):
        return self.answer
    