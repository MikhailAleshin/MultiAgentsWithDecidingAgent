def add_reasoning(sys_prompt, prompt_to_solve, row, cot=None):
    prompt_to_solve_add = prompt_to_solve.format(row=row)
    if cot:
        return '\n'.join([sys_prompt,
                          cot,
                          prompt_to_solve_add])
    else:
        return '\n'.join([sys_prompt,
                          prompt_to_solve_add])