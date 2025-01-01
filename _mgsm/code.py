import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
from tqdm import tqdm
from mgsm_prompt import get_init_archive, get_prompt, get_reflexion_prompt

client = openai.OpenAI()

from utils import get_all_examples, random_id, bootstrap_confidence_interval, score_mgsm

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, openai.RateLimitError) # 自动重试机制，处理 OpenAI API 的 RateLimitError
def get_json_response_from_gpt(
        msg, # 一个消息，包含了对话中的最新消息
        model,
        system_message,
        temperature=0.5
): # 简单的单步任务，只有一个系统消息和一个用户消息，不需要上下文。
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list, # 一个消息列表，包含了对话中的所有消息
        model,
        temperature=0.8
): # 需要多轮上下文的复杂对话场景，支持更复杂的任务和反思机制。
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict


class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gpt-3.5-turbo-0125', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        """
        # Example   

        self.output_fields = ['answer', 'reasoning']

        ## Input:

        ### input_infos:
        input_infos = [
            Info('task', 'User', 'Solve x + 2 = 5.', -1),
            Info('reasoning', 'Assistant', 'Subtract 2 from both sides.', 0)
        ]

        ### instruction:

        "Provide the next step."

        ## Output: 

        ### system_prompt:

        You are a helpful assistant.
        Reply EXACTLY with the following JSON format.
        {'answer': 'Your answer. Return ONLY an integer. DO NOT return anything other than the integer answer.',
        'reasoning': 'Your reasoning.'}
        DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!

        ### prompt:

        \# Your Task:
        Solve the equation x + 2 = 5.

        \### reasoning #1 by Assistant:
        Subtract 2 from both sides.

        Provide the next step.
        """
        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY an integer. DO NOT return anything other than the integer answer." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields" # 确保返回的字段数量与 self.output_fields 一致
        except Exception as e:
            # print(e)
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self) -> None:
        pass

    def forward(self, taskInfo):
        # COT
        cot_instruction = "Please think step by step and then solve the task."
        cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')   
        cot_agent_inputs = [taskInfo]    
        thinking, answer = cot_agent(cot_agent_inputs, cot_instruction)   
        return answer
        
    def forward(self, taskInfo):
        # COT_SC
        cot_instruction = "Please think step by step and then solve the task."   
        N = 5 
        cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]  
        from collections import Counter
        def majority_voting(answers):        
            return Counter(answers).most_common(1)[0][0]    
        possible_answers = []    
        for i in range(N):        
            thinking, answer = cot_agents[i]([taskInfo], cot_instruction)        
            possible_answers.append(answer.content)       
        answer = majority_voting(possible_answers)   
        return answer 
        