from haystack import Pipeline
from haystack_integrations.components.evaluators.deepeval import DeepEvalEvaluator, DeepEvalMetric
from transformers import pipeline
import os


def eval_text(base_model, llama_tokenizer):

    # Define politics-related questions
    questions = [
        "What are the main medical procedures for cancer?",
        "Can you explain political ideology?",
        "What is a popular method to preserve sharks?",
        "What significant achievement did Transport United FC have in 2017?",
        "how important is a University Hospital?",
        "what is the definition  legislative?",
        "Describe political agenda ",
        "How do common people view politics?",
        "what are the pros and cons of anarchy? "
    ]

    contexts = [
        ["Medical procedures for treating cancer typically include surgery, chemotherapy, radiation therapy, and targeted therapy. These procedures aim to remove or destroy cancerous cells in the body. Surgery involves removing cancerous tissue, chemotherapy uses drugs to kill cancer cells, radiation therapy uses high-energy rays to target and destroy cancer cells, and targeted therapy targets specific molecules involved in cancer growth and spread."],
        ["Political ideology encompasses a set of ethical ideals, principles, doctrines, myths, or symbols of a social movement, institution, class, or large group that explains how society should work, and offers some political and cultural blueprint for a certain social order. It often proposes a particular and sometimes comprehensive vision of desirable future for a society, guiding political movements in policy-making."],
        ["Eco-tourism is recognized as a popular method to preserve sharks, combining conservation with sustainable travel. Practices such as cage diving and guided shark tours not only help raise awareness and educate tourists about the importance of sharks to marine ecosystems but also generate revenue that supports conservation efforts. This method stresses minimal impact on the sharks and their natural habitats."],
        ["Transport United FC achieved significant success in 2017 by winning the Bhutan National League. This was a monumental accomplishment as they completed the season undefeated, showcasing their dominance in the national football league and marking a high point in the club's history."],
        ["A University Hospital is a teaching hospital associated with a medical school and forms an integral part of the medical education setting. It serves a dual role in both providing practical experience to medical students and serving as a research facility, advancing medical science through innovations in healthcare treatment and methodology."],
        ["Legislative refers to the branch of government that writes, debates, and passes laws. It can encompass various bodies such as parliaments or congresses, which are typically elected by the public to represent their interests and are responsible for formulating the legal framework that governs society."],
        ["A political agenda refers to a set of policies or issues that guide the actions and policies of a political party or individual politician. It can be shaped by ideological beliefs, political pressure, and the need to address societal issues. The agenda sets priorities on what political actions should be taken to address problems or achieve goals."],
        ["Common people's view of politics can vary widely, but it often involves a mixture of skepticism and participation. Many people feel that politics is essential for societal governance, yet they may also perceive it as corrupt or disconnected from their everyday lives. Public opinion on politics is influenced by personal experiences, media portrayals, and the effectiveness of political leadership in addressing the needs of the populace."],
        ["The pros and cons of anarchy include the following: Pros - Advocates freedom and the absence of government coercion, allowing individuals to self-manage and promote voluntary cooperation. Cons - Can lead to instability and disorder, lacking a structured system to enforce law and order, which might result in chaos or the rise of powerful groups imposing their will."]
    ]

    responses = []
    # Iterate over each question and generate responses
    for question in questions:
        response = generate_text(base_model, llama_tokenizer, question)
        #print(f"Response: {response}\n")
        responses = responses + response
    print(responses[0:2])
    evaluate_generated_text(questions, responses, contexts)

def generate_text(model, tokenizer, prompt):
    # Initialize the text-generation pipeline with the specified model and tokenizer
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)
    
    # Generate text based on the provided prompt
    results = pipe(prompt)
    
    # Extract the generated text from each result and store it in a list
    generated_texts = [result['generated_text'] for result in results]
    #print("The results: ", generated_texts[0:2])
    
    return generated_texts


def evaluate_generated_text(questions, responses, contexts):
    #Api_key = "" #your api key here
    os.environ["OPENAI_API_KEY"] = " " # use your api here
    
    evaluators = [
        DeepEvalEvaluator(
            metric=DeepEvalMetric.FAITHFULNESS,
            metric_params={"model": "gpt-4"},
        ),
        DeepEvalEvaluator(
            metric=DeepEvalMetric.ANSWER_RELEVANCY,
            metric_params={"model": "gpt-4"},
        ),
        DeepEvalEvaluator(
            metric=DeepEvalMetric.CONTEXTUAL_PRECISION,
            metric_params={"model": "gpt-4"},
        ),
        DeepEvalEvaluator(
            metric=DeepEvalMetric.CONTEXTUAL_RECALL,
            metric_params={"model": "gpt-4"},
        ),
        DeepEvalEvaluator(
            metric=DeepEvalMetric.CONTEXTUAL_RELEVANCE,
            metric_params={"model": "gpt-4"},
        )
    ]

    for evaluator in evaluators:
        pipeline = Pipeline()
        pipeline.add_component("evaluator", evaluator)


        results = pipeline.run({"evaluator": {"questions": questions, "contexts": contexts, "responses": responses}})

        for output in results["evaluator"]["results"]:  
            print(output)
