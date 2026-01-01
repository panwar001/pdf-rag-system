
from get_llm import get_llm
from ingest_pipeline import DB_PATH, load_vector_store
from langchain.prompts import ChatPromptTemplate
from user_prompts import USER_PROMPT


# RETRIEVE CHUNKS
def retrieve_chunks(query: str, k: int = 5):
    vector_store = load_vector_store()

    docs = vector_store.similarity_search(
        query=query,
        k=k #can be removed from here
        # kwargs={
        #     "k": k,
        #     "filter": {"policy_scope": "international_only"}
        # }
    )

    return docs

def build_context_and_citations(docs):
    context_parts = []
    citations = []

    for doc in docs:
        context_parts.append(doc.page_content)

        citation = (
            f"[source={doc.metadata.get('source', 'unknown')} | "
            f"page={doc.metadata.get('page', 'N/A')}]"
        )
        citations.append(citation)

    context = "\n\n".join(context_parts)
    citations = list(set(citations))  # remove duplicates

    return context, citations


def answer_question_from_pdf(question: str):
    # Step 1: Retrieve
    print("Query is... ", {question} )
    docs = retrieve_chunks(question, k=5)
    # if not docs:
    #     return {
    #         "answer": "No relevant information found in the document.",
    #         "citations": [],
    #         "confidence": 0.0
    #     }
    # Step 2: Build context
    context, citations = build_context_and_citations(docs)

    # Step 3: Prompt
    prompt = USER_PROMPT.format(
        context=context,
        question=question
    )

    # Step 4: LLM call
    llm = get_llm()
    response = llm.invoke(prompt)
    #response = ask_open_source_model(prompt)

    # Step 5: Confidence (simple & safe)
    confidence = round(min(len(docs) / 5, 1.0), 2)

    print("Answer---------------:\n", response.content)
   # return response
    return {
        "answer": response.content,
        "citations": citations,
        "confidence": confidence
    }

if __name__ == "__main__":
    #query = "Is dental treatment covered during international travel?"
    #query = "Is dental treatment covered during domestic travel?"
    query = "Is dental treatment covered during domestic travel in Tripsafe Plan?"
    #query = "Can I buy Tripsafe for domestic travel?"
    #query = "Is Knee Surgery covered during international travel?"
    #query = "Is Knee Surgery covered during domestic travel?" #answer coming incorrect as policy is for international travel

    print(f"User : - {query}")
    result = answer_question_from_pdf(query)

    print("Answer:\n", result["answer"])
    print("\nCitations:")
    for c in result["citations"]:
        print("-", c)
    print("\nConfidence:", result["confidence"])


# def main():
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#     query_rag(query_text)
#
#
# def query_rag(query_text: str):
#     # Prepare the DB.
#     # embedding_function = get_embedding_function()
#     # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
#     db = Chroma(persist_directory=DB_PATH)
#     # Search the DB.
#     results = db.similarity_search_with_score(query_text, k=5)
#
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#     # print(prompt)
#
#     model = Ollama(model="mistral")
#     response_text = model.invoke(prompt)
#
#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)
#     return response_text
#
#
# if __name__ == "__main__":
#     main()