
import numpy as np
import csv

from models import DINA, DINO, ACDM
from estimate import MLE


if __name__ == "__main__":
    model = ACDM(
        slip=np.array([0.2] * 12),
        guess=np.array([0.4] * 12)
    )

    with open("data/q_matrix.csv", "r") as f:
        reader = csv.reader(f)
        skill_name = next(reader)[1:]
        qm = np.array([[int(i) for i in row[1:]] for row in reader])
        print("Q-matrix Loaded: ", qm.shape)
    
    with open("data/student_answers.csv", "r") as f:
        reader = csv.reader(f)
        student_id = next(reader)[1:]
        x = np.array([[int(i) for i in row[1:]] for row in reader])
        print("Student Answer Loaded: ", x.shape)
    
    alpha = MLE(model, qm, x)

    for i in range(len(student_id)):
        print(student_id[i], ":", [skill_name[j] for j in range(len(skill_name)) if alpha[i][j] == 1])

