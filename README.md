<div align="center">

# MicroLlama-300M

</div>

As an individual with limited access and compute, I have been wondering if I could build a decent large-language model for a while. As the big mega corporations are focused on getting bigger and bigger models, I am going small! 

<div align="center">
  <img src="./microllama.jpg" width="300"/>
</div>

As a result, I set up the following goals to **pretraining** a **300M Llama model** with the following restrictions:

1. My overall budget is $500.
2. Must pretrain an LLM from scratch with a fully open-source dataset and model.
3. Not allowed to finetune a model or use another LLM such as GPT-4 to generate any training data.

This project is heavily based on [TinyLlama](https://github.com/jzhang38/TinyLlama), which is an awesome open-source project aimed to **pretraining** a **1.1.1B Llama model on 1T tokens**. 

This project is work in progress. Currently, I have spent \$280 on compute using 4 x Nvidia 4090 on [Vast.ai](https://vast.ai) and \$3 on AWS S3 storage after 4 days of training of the **300M Llama model** with **50B** tokens.

I modified [TinyLlama](https://github.com/jzhang38/TinyLlama) to support the following features (I will release my forked version of the source code after some clean up):
1. Pretrain a smaller size 300M model on [Slimpajama](https://huggingface.co/datasets/cerebras/slimpajama-627b)
2. Removed [Starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) so that my model can focus on [Slimpajama](https://huggingface.co/datasets/cerebras/slimpajama-627b). This also means my model probably cannot do coding without fine-tuning.
3. Added the ability to process and tokenize [Slimpajama](https://huggingface.co/datasets/cerebras/slimpajama-627b) while downloading the data. The original setup only works with pre-downloaded data. This turns out to be a good time-saver because downloading 800G+ of data on a non-commercial Internet is very slow, and processing all of [Slimpajama](https://huggingface.co/datasets/cerebras/slimpajama-627b) data also takes time.
4. Various helper scripts and Python code such as python code for uploading the pretrained checkpoint to the huggingface hub.
5. Bug fixes.

## News and Updates

**08/10/2025** - Released [MicroLaVa](https://github.com/keeeeenw/MicroLlava), a lightweight Visual Q&A model based on [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory). The [MicroLaVa](https://github.com/keeeeenw/MicroLlava) model contains only 700M parameters in total and can be pre-trained and fine-tuned on a single Nvidia RTX 4090. Built on MicroLlama, it already produces meaningful results with further improvements planned.

**11/10/2024** - Published [MicroLlama-text-embedding](https://huggingface.co/keeeeenw/MicroLlama-text-embedding/), a compact text embedding model for sentence similarity tasks. Based on MicroLlama architecture, this model is in active development and requires further optimization.

**06/04/2024** - MicroLlama became officially supported in [LitGPT](https://github.com/Lightning-AI/litgpt). Thanks to the Lightning AI team for accepting my [merge request](https://github.com/Lightning-AI/litgpt/pull/1457) and providing Lightning AI Studio credits.


## Evaluation results

I performed the experiment using the standard [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) setup. Following the same setup as [TinyLlama](https://github.com/jzhang38/TinyLlama), I used **acc_norm** for all datasets except for **winogrande** and **boolq** which used **acc** as the metrics.

1. **[keeeeenw/MicroLlama](https://huggingface.co/keeeeenw/MicroLlama)** is the evaluation results for my **300M Llama model on 50B tokens**.
2. **[google-best/bert-large-uncased](https://huggingface.co/google-bert/bert-large-uncased)** is the baseline because it is one of the most popular small LLMs and it has a similar parameter count of **336M**.
3. **[PY007/TinyLlama-1.1B-Chat-v0.1](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.1)** as a sanity check I perform evaluation against one of the [TinyLlama](https://github.com/jzhang38/TinyLlama) models to validate my setup for [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). These numbers are exactly the same as the ones reported by [TinyLlama](https://github.com/jzhang38/TinyLlama).
4. **TinyLlama-1.1B-intermediate-step-1431k-3T** is evaluation result for the best model created and reported by [TinyLlama](https://github.com/jzhang38/TinyLlama).

| Model                                      | Pretrain Tokens | HellaSwag | Obqa  | WinoGrande | ARC_c | ARC_e | boolq | piqa  | avg   |
|--------------------------------------------|-----------------|-----------|-------|------------|-------|-------|-------|-------|-------|
| keeeeenw/MicroLlama                        | 50B             | 34.30     | 30.60 | 51.54      | 23.29 | 39.06 | 53.15 | 64.58 | 42.36 |
| google-best/bert-large-uncased             | N/A             | 24.53     | 26.20 | 49.80      | 25.68 | 25.08 | 40.86 | 47.66 | 34.26 |
| PY007/TinyLlama-1.1B-Chat-v0.1             | 503B            | 53.81     | 32.20 | 55.01      | 28.67 | 49.62 | 58.04 | 69.64 | 49.57 |
| TinyLlama-1.1B-intermediate-step-1431k-3T  | 3T              | 59.20     | 36.00 | 59.12      | 30.12 | 55.25 | 57.83 | 73.29 | 52.99 |

To reproduce my numbers, please install [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and run the following command:
```bash
lm_eval \
    --model hf \
    --model_args pretrained=keeeeenw/MicroLlama,dtype="float",tokenizer=TinyLlama/TinyLlama-1.1B-step-50K-105b \
    --tasks hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa \
    --device cuda:0 \
    --batch_size 64
```

#### Observations
1. Because [keeeeenw/MicroLlama](https://huggingface.co/keeeeenw/MicroLlama) is much smaller than [TinyLlama](https://github.com/jzhang38/TinyLlama), our model does not achieve the same impressive results but the numbers are closer than I expected.
2. Our model outperforms [google-best/bert-large-uncased](https://huggingface.co/google-bert/bert-large-uncased) which is actually slightly larger. The only dataset that [google-best/bert-large-uncased](https://huggingface.co/google-bert/bert-large-uncased) outperformed our model is ARC_c (arc_challenge). I will provide more analysis as future study.

Based on the evaluation above, our model should be a good starting point for fine-tunning tasks that are typically performed using the BERT family of models. Some of tasks may include
1. (sentence transformer)[https://huggingface.co/sentence-transformers], 
2. (bertscore)[https://huggingface.co/spaces/evaluate-metric/bertscore]
3. A light-weight chatbot after some finetuning.

#### Want to try it out?

1. Install dependencies
```
pip install transformers
```
2. Run code!

```python
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

def generate_text(prompt, model, tokenizer):
    text_generator = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        tokenizer=tokenizer
    )

    formatted_prompt = f"Question: {prompt} Answer:"

    sequences = text_generator(
        formatted_prompt,
        do_sample=True,
        top_k=5,
        top_p=0.9,
        num_return_sequences=1,
        repetition_penalty=1.5,
        max_new_tokens=128,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

# use the same tokenizer as TinyLlama
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b")

# load model from huggingface
# question from https://www.reddit.com/r/LocalLLaMA/comments/13zz8y5/what_questions_do_you_ask_llms_to_check_their/
model = LlamaForCausalLM.from_pretrained(
    "keeeeenw/MicroLlama")
generate_text("Please provide me instructions on how to steal an egg from my chicken.", model, tokenizer)
```

## Acknowledgements
This repository is built upon [TinyLlama](https://github.com/jzhang38/TinyLlama) which is based on [lit-gpt](https://github.com/Lightning-AI/lit-gpt) and [flash-attention](https://github.com/Dao-AILab/flash-attention).
```
@misc{zhang2024tinyllama,
      title={TinyLlama: An Open-Source Small Language Model}, 
      author={Peiyuan Zhang and Guangtao Zeng and Tianduo Wang and Wei Lu},
      year={2024},
      eprint={2401.02385},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@online{lit-gpt,
  author    = {Lightning AI},
  title     = {Lit-GPT},
  url       = {https://github.com/Lightning-AI/lit-gpt},
  year      = {2023},
}
@article{dao2023flashattention2,
  title     ={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author    ={Dao, Tri},
  year      ={2023}
}
```

Special thanks to Xinyang, creator of the outstanding [OpenLLaMA](https://github.com/openlm-research/open_llama) project, whose work inspired me to embark on this project. I’m grateful for the guidance in foundational concepts such as establishing scaling laws before training a model.

## Citation
If you use MicroLlama in your research or work, please cite the project using the following reference:
APA:
```
Wang, Z. K. (2024). MicroLlama: A 300M-parameter language model trained from scratch. GitHub & Hugging Face. https://github.com/keeeeenw/MicroLlama, https://huggingface.co/keeeeenw/MicroLlama
```
BibTeX:
```
@misc{wang2024microllama,
  author       = {Zixiao Ken Wang},
  title        = {MicroLlama: A 300M-parameter language model trained from scratch},
  year         = {2024},
  howpublished = {\url{https://github.com/keeeeenw/MicroLlama}, \url{https://huggingface.co/keeeeenw/MicroLlama}},
  note         = {GitHub and Hugging Face repositories}
}
```
🙏 Please cite this work if you find it useful.
