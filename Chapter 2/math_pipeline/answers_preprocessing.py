def answers_to_list(data_llm, i):
    return list(set(data_llm[data_llm.columns].iloc[i].values))

def generate_test(data_llm,
                   i,
                   test,
                   final_agent_cot,
                   test_list = ['A', 'B', 'C', 'D', 'E',
                                'F', 'G', 'H', 'I', 'G',
                                'K', 'L', 'M', 'N', 'O'], **kwargs):
    question = test[i]['question']
    answers_list = answers_to_list(data_llm, i)
    answers_dict = {test_list[i]: answers_list[i] for i in range(len(answers_list))}
    n = '\n'
    return f"""{final_agent_cot}QUESTION: {question}{n}ANSWERS:{n}{''.join([f'{key}) {value}{n}' for key, value in answers_dict.items()])}{n}Your reasoning:{n}"""