import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import data_preprocessing
import dataset
import dataloader
import scoring_agents
import answers_preprocessing
from torch.utils.data import DataLoader

def load_model(device):
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    return model, tokenizer

def start_experiment_agents_scoring(yaml_path):
    with open(yaml_path, 'rt') as f:
        conf = yaml.safe_load(f)
    model, tokenizer = load_model(conf['device'])
    data = dataloader.DataLoaderExperiment(data_prepor=data_preprocessing.add_reasoning, **conf)
    data.read_data()
    data.generate_cot(**conf)
    data.generate_final_questions(**conf)
    list_questions = dataset.MyDataset([element['question'] for element in data.test_form])
    answers = [int(element['final_ans']) for element in data.test_form]
    dataloader_exp = DataLoader(list_questions, batch_size = 5)
    agents_answers = scoring_agents.start_scoring(dataloader_exp, tokenizer, model, **conf)
    df_answers, df_bool = scoring_agents.parse_and_save_all_agents(
                                            agents_answers,
                                            answers,
                                            conf['agents_answers_path_to_save'],
                                            conf['agents_bool_path_to_save']
                                            )
    return conf, model, tokenizer, data, answers, df_answers, df_bool


def start_experiment_final_agent(
                                 conf,
                                 model,
                                 tokenizer,
                                 data,
                                 answers,
                                 df_answers
                                 ):
    formated_answers = [
                        answers_preprocessing.generate_test(
                        df_answers, i, data.test, **conf
                        ) for i in range(len(data.test))
                       ]
    final_agent_answers = scoring_agents.scoring_and_parsing_final_agent(
                                    tokenizer,
                                    model,
                                    conf['final_agent_config'],
                                    df_answers,
                                    formated_answers,
                                    conf['device'],
                                    conf['path_to_save_final_agent'],
                                    conf['test_list']
    )
    return final_agent_answers, answers, data

def start_all(yaml_path):
    conf, model, tokenizer, data, answers, df_answers, df_bool = start_experiment_agents_scoring(yaml_path)
    final_agent_answers, answers, data = start_experiment_final_agent(
                                                                        conf,
                                                                        model,
                                                                        tokenizer,
                                                                        data,
                                                                        answers,
                                                                        df_answers
    )
    return final_agent_answers, answers, data



    