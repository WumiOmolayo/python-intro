from quiz_data import get_questions
import random, datetime

question_bank = get_questions()

#print(question_bank)

random.shuffle(question_bank)
print("=" * 60)
print(" ")
print(question_bank)


questions_only = []
answers_only = []


for question in question_bank:
    # print(question[0])
    questions_only.append(question[0])
    answers_only.append(question[1])


print(questions_only)
print(answers_only)