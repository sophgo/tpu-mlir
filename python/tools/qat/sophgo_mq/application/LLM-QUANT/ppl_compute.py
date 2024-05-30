import evaluate
from evaluate import logging
from datasets import load_dataset
perplexity = evaluate.load("perplexity", module_type="metric")
input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")["text"] 
#input_texts = load_dataset("zhengxuanzenwu/wikitext-2-split-128",split="test")['text']
input_texts = [s for s in input_texts if s!='']
results = perplexity.compute(model_id='Aalaa/opt-125m-wikitext2',
                            predictions=input_texts)
print(list(results.keys()))
print(round(results["mean_perplexity"], 2))
#print(results["perplexities"])