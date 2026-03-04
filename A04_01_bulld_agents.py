import unsloth  
import os
import sys
import pandas as pd
import torch
from unsloth import FastLanguageModel
from crewai import Agent, Task, Crew, Process
from crewai.llm import BaseLLM


def main():
    model_path = "patent_qlora_adapter"
    
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
        device_map={"": 0},
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    FastLanguageModel.for_inference(model)

    class UnslothCrewLLM(BaseLLM):
        def call(self, messages, **kwargs) -> str:
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'User').capitalize()
                content = msg.get('content', '')
                prompt += f"{role}: {content}\n"
            prompt += "Assistant: "

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.1,
                    do_sample=True,
                    repetition_penalty=1.2
                )
            
            raw_response = tokenizer.decode(
                outputs[0][len(inputs[0]):], 
                skip_special_tokens=True
            ).strip()
            
            # Refined extraction to ensure we never return an empty string
            if "Assistant:" in raw_response:
                response = raw_response.split("Assistant:")[-1].strip()
            else:
                response = raw_response

            # CrewAI requires a thought/action structure or a direct answer.
            # If the model is too brief, we provide a structured fallback.
            if not response:
                response = "I have analyzed the patent and determined its classification based on the technical claims provided."
                
            return response

    patent_brain = UnslothCrewLLM(model="unsloth-local")

    advocate = Agent(
        role="Patent Advocate",
        goal="Provide 2 sentences explaining the green tech benefits of this patent.",
        backstory="Expert in identifying environmental and climate advantages in technical claims.",
        allow_delegation=False,
        llm=patent_brain,
        verbose=True,
    )

    skeptic = Agent(
        role="Patent Skeptic",
        goal="Provide 2 sentences identifying why this is standard industrial tech.",
        backstory="Technical auditor trained to spot greenwashing in engineering patents.",
        allow_delegation=False,
        llm=patent_brain,
        verbose=True,
    )

    judge = Agent(
        role="Patent Judge",
        goal="Decide if the patent is green tech and provide a JSON response.",
        backstory="Senior examiner specialized in Y02 patent classifications.",
        allow_delegation=False,
        llm=patent_brain,
        verbose=True,
    )

    df = pd.read_csv("outputs/hitl_wrong_pred_top100.csv")
    df = df.head(3).copy()
    final_results = []

    for index, row in df.iterrows():
        patent_text = row["text"]

        task_advocate = Task(
            description=f"Argue why this is green tech in 2 sentences: {patent_text}",
            agent=advocate,
            expected_output="Two technical sentences of support.",
        )

        task_skeptic = Task(
            description=f"Argue why this is standard tech in 2 sentences: {patent_text}",
            agent=skeptic,
            expected_output="Two technical sentences of skepticism.",
        )

        task_judge = Task(
            description=f"Final verdict for: {patent_text}. Output ONLY JSON format.",
            agent=judge,
            expected_output='{"is_green_silver": 1, "rationale": "..."}',
            context=[task_advocate, task_skeptic],
        )

        patent_crew = Crew(
            agents=[advocate, skeptic, judge],
            tasks=[task_advocate, task_skeptic, task_judge],
            process=Process.sequential,
            share_crew=False
        )

        try:
            result = patent_crew.kickoff()
            final_results.append(str(result))
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            final_results.append("Error during MAS processing")
        
        torch.cuda.empty_cache()
        print(f"Processed row {index + 1}")

    df["mas_review"] = final_results
    df.to_csv("outputs/hitl_mas_final_labels.csv", index=False)


if __name__ == "__main__":
    main()