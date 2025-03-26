from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ResponseGenerator:
    def __init__(
        self, 
        model_name: str = "google/flan-t5-small",
        temperature: float = 0.3
    ):
        """
        Initialize the response generator with a Hugging Face model.
        
        Args:
            model_name (str): Name of the Hugging Face model
            temperature (float): Sampling temperature for text generation
        """
        try:
            self.llm = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text2text-generation",
                model_kwargs={
                    "temperature": temperature,
                    "max_length": 500
                }
            )
            self.output_parser = StrOutputParser()
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def generate_response(
        self, 
        query: str, 
        context: str
    ) -> str:
        """
        Generate a response using the language model and retrieved context.
        
        Args:
            query (str): User's query
            context (str): Retrieved context from documents
        
        Returns:
            str: Generated response
        """
        try:
            prompt_template = (
                "Context:\n{context}\n\n"
                "Based on the context, answer the following query:\n"
                "Query: {query}\n\n"
                "Answer:"
            )

            prompt = ChatPromptTemplate.from_template(prompt_template)

            chain = prompt | self.llm | self.output_parser
            
            response = chain.invoke({
                "context": context,
                "query": query
            })
            
            return response
        
        except Exception as e:
            return f"Error generating response: {e}"