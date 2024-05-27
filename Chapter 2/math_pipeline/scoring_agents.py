import itertools
from transformers import GenerationConfig
from tqdm import tqdm
import re
import pandas as pd
import answers_preprocessing

def config_experiment(
                      dataloader,
                      config,
                      device,
                      tokenizer,
                      model
                      ):
    answer = []
    for questions in tqdm(dataloader):
        input_ids = tokenizer.batch_encode_plus(
                    questions,
                    return_tensors='pt',
                    padding=True
                )['input_ids'].to(device)
        answer.append(tokenizer.batch_decode(
                    model.generate(
                            input_ids,
                            generation_config=config,
                        pad_token_id=2
                    )[:, input_ids.shape[1]:],
                    skip_special_tokens=True))
    return list(itertools.chain(*answer))

def start_scoring( 
                  dataloader,
                  tokenizer,
                  model,
                  config_list,
                  device,
                  **kwargs
                  ):
    agents_answer = []
    for config in config_list:
        config = GenerationConfig.from_dict(config)
        agents_answer.append(config_experiment(dataloader,
                                               config,
                                               device,
                                               tokenizer,
                                               model)
                            )
    return agents_answer

def parse_agent_answer(agent_answers,
                       true_answers):
    llm_answer = []
    bool_result = []
    for row_num in range(len(agent_answers)):
        try:
            llm_answer.append(int(re.findall('\d+',
                                    agent_answers[row_num].split(
                                        'My answer:'
                                        )[1].split('\n')[0].replace(' ', ''))[0]))
        except:
            llm_answer.append('-9999')
        bool_result.append(llm_answer[row_num] == true_answers[row_num])
            
    return llm_answer, bool_result

def parse_and_save_all_agents(
                     agents_answers,
                     true_answers,
                     agents_answers_path_to_save,
                     agents_bool_path_to_save
                    ):
    answers_list = []
    bool_answers_list = []
    for single_answers in agents_answers:
        int_answers, bool_answers = parse_agent_answer(single_answers, true_answers)
        answers_list.append(int_answers)
        bool_answers_list.append(bool_answers)
    data_bool = pd.DataFrame({f'agent_{i}': bool_answers_list[i] for i in range(len(bool_answers_list))})
    data_llm = pd.DataFrame({f'agent_{i}': answers_list[i] for i in range(len(answers_list))})
    print(data_bool.mean(), f'Max possible accuracy {data_bool.max(axis=1).mean()}')
    data_bool.to_csv(agents_bool_path_to_save, index=False)
    data_llm.to_csv(agents_answers_path_to_save, index=False)
    return data_llm, data_bool


def generate_answer(question, model, tokenizer, cfg, device, **kwargs):
    input_ids = tokenizer.encode(question, return_tensors='pt').to(device)
    return tokenizer.decode(
                model.generate(input_ids,
                        generation_config=cfg, pad_token_id=2)[0][input_ids.shape[1]:])


def scoring_and_parsing_final_agent(
                                    tokenizer,
                                    model,
                                    final_agent_config,
                                    df_answers,
                                    quesions,
                                    device,
                                    path_to_save,
                                    test_list,
                                    **kwargs
                                    ):
    final_agent_answers = []
    answers_dict = []
    cfg = GenerationConfig.from_dict(final_agent_config[0])
    single_flag = []
    for j in range(len(quesions)):
        list_ = answers_preprocessing.answers_to_list(df_answers, j)
        if len(list_) == 1:
            single_flag.append(True)
        else: 
            single_flag.append(False)
        answers_dict.append({test_list[i]: list_[i] for i in range(len(list_))})
    for i in tqdm(range(len(quesions))):
        if single_flag[i]:
            final_agent_answers.append(answers_dict[i][test_list[0]])
        else:
            answer = generate_answer(quesions[i], model, tokenizer, cfg, device)
            try:
                answ_letter = answer.split('My answer:')[1].split('\n')[0].replace(' ', '').replace('.', '').replace('</s>', '')
            except:
                answ_letter = test_list[0]
            number = answers_dict[i][answ_letter[0]]
            final_agent_answers.append(number)
    final_agent_answers = pd.DataFrame({'final_agent': final_agent_answers})
    final_agent_answers.to_csv(path_to_save)
    return final_agent_answers

