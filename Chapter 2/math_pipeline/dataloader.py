import json
from typing import Callable

class DataLoaderExperiment():
    def __init__(
                self,
                train_path: str,
                test_path: str,
                data_prepor: str,
                **kwargs
                ):
        self.train_path = train_path
        self.test_path = test_path
        self.data_prepor = data_prepor

    def json_load(self,
                  path: str):
        with open(path, 'r') as f:
            data = json.load(f)    
        return data
    
    def read_data(self):
        self.train, self.test = [self.json_load(path) for path
                                 in [self.train_path, self.test_path]]
    
    def generate_cot(
                     self,
                     cot_list,
                     cot_train_list,
                     make_reasoning_cot,
                    **kwargs
                    ):
        self.cot = '\n'.join(
            [make_reasoning_cot.format(
                                question=self.train[index]['question'],
                                cot=cot,
                                answer=self.train[index]['final_ans'])
                              for index, cot in zip(cot_train_list, cot_list)]
        )

    def generate_final_questions(
                                 self,
                                 sys_prompt,
                                 prompt_to_solve,
                                **kwargs
                                ):
        self.test_form = self.test.copy()
        if self.cot:
            for i in range(len(self.test_form)):
                self.test_form[i]['question'] = self.data_prepor(
                                                                 sys_prompt,
                                                                 prompt_to_solve,
                                                                 self.test[i]['question'],
                                                                 self.cot
                )
        else: 
            for i in range(len(self.test_form)):
                self.test_form[i]['question'] = self.data_prepor(
                                                                 sys_prompt,
                                                                 prompt_to_solve,
                                                                 self.test[i]['question']
                )



