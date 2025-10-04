import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import List, Any, Dict, Optional

import dashscope
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

from llm_config import (
    EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, LLM_MODEL_PATH, DEVICE,
    EMBEDDING_MAX_LENGTH, RERANKER_MAX_LENGTH, LLM_MAX_NEW_TOKENS
)

def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Helper function to perform last token pooling.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class QwenEmbeddings_local(Embeddings):
    model: Any = None
    tokenizer: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL_PATH, trust_remote_code=True, padding_side='left'
        )
        self.model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16
        ).to(DEVICE).eval()

    def _get_instruct(self, task_description: str, query: str) -> str:

        return f'Instruct: {task_description}\nQuery: {query}'
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            batch_dict = self.tokenizer(
                texts, padding=True, truncation=True, 
                max_length=EMBEDDING_MAX_LENGTH, return_tensors="pt"
            )
            batch_dict = {k: v.to(DEVICE) for k, v in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = _last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            return normalized_embeddings.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:

        task = "Given a web search query, retrieve relevant passages that answer the query"
        instructed_text = self._get_instruct(task, text)
        embedding = self._embed([instructed_text])
        return embedding[0]



class QwenEmbeddings(Embeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dashscope.api_key = kwargs.get('api_key', '')  # 从参数获取API key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = dashscope.TextEmbedding.call(
            model='text-embedding-v3',
            input=texts,
            text_type='document'
        )
        return [item['embedding'] for item in response.output['embeddings']]

    def embed_query(self, text: str) -> List[float]:
        response = dashscope.TextEmbedding.call(
            model='text-embedding-v3',
            input=text,
            text_type='query'
        )
        return response.output['embeddings'][0]['embedding']



class QwenReranker(BaseDocumentCompressor):

    model: Any = None
    tokenizer: Any = None
    top_n: int = 5
    token_false_id: int = 0
    token_true_id: int = 0
    prefix_tokens: List[int] = None
    suffix_tokens: List[int] = None

    def __init__(self, top_n: int, **kwargs):
        super().__init__(**kwargs)
        self.top_n = top_n

        self.tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL_PATH, trust_remote_code=True, padding_side='left'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)


    def _format_instruction(self, instruction, query, doc):
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:

        if not documents:
            return []
        
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        pairs = [self._format_instruction(instruction, query, doc.page_content) for doc in documents]
        
        with torch.no_grad():

            inputs = self.tokenizer(
                pairs, padding=False, truncation='longest_first',
                return_attention_mask=False, max_length=RERANKER_MAX_LENGTH - len(self.prefix_tokens) - len(self.suffix_tokens)
            )

            for i in range(len(inputs['input_ids'])):
                inputs['input_ids'][i] = self.prefix_tokens + inputs['input_ids'][i] + self.suffix_tokens

            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=RERANKER_MAX_LENGTH)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            
            stacked_scores = torch.stack([false_vector, true_vector], dim=1)
            log_probs = torch.nn.functional.log_softmax(stacked_scores, dim=1)
            scores = log_probs[:, 1].exp().cpu().tolist()
        

        for doc, score in zip(documents, scores):
            doc.metadata['rerank_score'] = score
        
        sorted_docs = sorted(documents, key=lambda x: x.metadata['rerank_score'], reverse=True)
        return sorted_docs[:self.top_n]

class HttpsApi:
    def __init__(self, model: LLMConfig, **kwargs):
        """
        Initialize the HttpsApi class.

        :param host: The host of the API.
        :param key: The API key.
        :param model: The model to use.
        :param url: The URL of the API.
        :param timeout: The timeout for the API request.
        :param kwargs: Additional keyword arguments.
        """
        self._host = host
        self._key = key
        self._model = model
        self._url = url
        self._timeout = timeout
        self._kwargs = kwargs
        self._max_retry = 10

    def get_response(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        retry = 0
        while True:
            try:
                if self._model.startswith("o1-preview"):
                    for p in prompt:
                        if p['role'] == 'system':
                            p['role'] = 'user'

                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps({
                    # 'max_tokens': self._kwargs.get('max_tokens', 4096),
                    # 'top_p': self._kwargs.get('top_p', None),
                    'temperature': self._kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': prompt
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request('POST', self._url, payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)
                response = data['choices'][0]['message']['content']
                # if self._model.startswith('claude'):
                #     response = data['content'][0]['text']
                # else:
                #     response = data['choices'][0]['message']['content']
                return response
            except Exception as e:
                retry += 1
                if retry >= self._max_retry:
                    raise RuntimeError(
                        # f'{self.__class__.__name__} error: {traceback.format_exc()}.\n'
                        f'Model Response Error! You may check your API host and API key.'
                    )
                else:
                    print(f'Model Response Error! Retrying...')
                    # print(f'{self.__class__.__name__} error: {traceback.format_exc()}. Retrying...\n')


class QwenLLM(LLM):
    model: Any = None
    tokenizer: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        ).eval()


    @property
    def _llm_type(self) -> str:
        return "qwen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(DEVICE)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            attention_mask=model_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        input_len = model_inputs.input_ids.shape[1]
        response_ids = generated_ids[0][input_len:]
        
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response