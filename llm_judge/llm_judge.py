from openai import OpenAI
import random
import openai
from tqdm import tqdm
correct_num=0
total=0

with open('posh/island-wh.txt') as posh, open('llm_island-wh_5.4.tsv', 'w') as result:
    result.write(f'correct\tincorrect\tmodel_choice\n')
    sentences = posh.readlines()
    for i in tqdm(range(0, 1000, 2)):
        try:
            correct = sentences[i+1].strip()
            incorrect = sentences[i].strip()
            correct_option = random.choice(['A', 'B'])

            options = ['A', 'B']
            options.remove(correct_option)
            incorrect_option = options[0]
            if correct_option == "A":
                sentence_prompt = f"Sentence A: {correct}\n Sentence B: {incorrect}"
            else:
                sentence_prompt = f"Sentence A: {incorrect}\n Sentence B: {correct}"

            prompt = f"""
                Which sentence is grammatical?
                
                {sentence_prompt}
                
                Respond with exactly one character: A or B.
                """


            response = client.responses.create(
                model="gpt-5.4-2026-03-05",
                # model="gpt-5.5-2026-04-23",
                input=prompt
            )

            answer = response.output_text.strip()
            if correct_option!=answer:
                print(sentence_prompt)
                print(correct_option, answer)

            result.write(f'{correct_option}:{correct}\t{incorrect_option}:{incorrect}\t{answer}\n')
            if answer==correct_option:
                correct_num+=1
            total+=1
        except openai.PermissionDeniedError:
            continue

print(correct_num/total)
