import json
import os
import asyncio

from raptor import RetrievalAugmentation
from tqdm.asyncio import tqdm_asyncio

async def run(samples, rac_config):
    
    print("Length of samples", len(samples))
    max_concurrent_tasks = 10
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
         
    async def sem_task(sample):
        async with semaphore:
            return await run_sample(sample, rac_config)
        
    # Create a list of tasks
    tasks = [asyncio.create_task(sem_task(sample)) for sample in samples]
    
    # Gather all results
    results = await tqdm_asyncio.gather(*tasks)
    return results
                
async def run_sample(sample, rac_config):
    doc = sample["title"]
    doc += "\n\n"
    doc += "abstract" + "\n"
    doc += sample["abstract"] + "\n\n"
    paragraphs = sample["full_text"]["paragraphs"]
    names = sample["full_text"]["section_name"]
    for name, paras in zip(names, paragraphs):
        doc += name + "\n"
        doc += "\n".join(paras)
        doc += "\n\n"
        
    doc += "figures_and_tables\n"
    doc += "\n".join(sample["figures_and_tables"]["caption"])
    # print(doc)
    RA = RetrievalAugmentation(config=rac_config)
    RA.add_documents(doc)
    results = []
    for idx in range(len(sample["qas"]["question"])):
        question_id = sample["qas"]["question_id"][idx]
        question = sample["qas"]["question"][idx]
        answer = RA.answer_question(question=question)
        print("Question: ", question)
        print("Answer: ", answer)
        print("\n\n")
        results.append({"question_id": question_id, "question": question, 
                "predicted_evidence": "", "predicted_answer":answer})
            
    paper_id = sample["id"]
    folder_path = "results/outputs"
    res_path = os.path.join(folder_path, f"{paper_id}.jsonl")
    with open(res_path, "w") as f:
        for res in results:
            json.dump(res, f)
            f.write("\n")
    return results