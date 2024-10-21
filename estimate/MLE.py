
import numpy as np

def MLE(model, qm, x):   # 학생의 인지요소를 추정
    skill_prob = []
    for i in range(2 ** qm.shape[1]):   # 인지요소의 모든 조합에 대해
        alpha = (i >> np.arange(qm.shape[1]) & 1)[np.newaxis, :]    # ndarray[1, 인지요소 id] -> 0 or 1 (인지요소 조합)
        skill_prob.append(model.L(alpha, qm, x))
    skill_prob = np.array(skill_prob)   # ndarray[2^인지요소 수, 학생 id]

    v = skill_prob.argmax(axis=0)    # ndarray[학생 id] -> 0~2^인지요소 수 (가장 가능성 높은 인지요소 조합)

    prediction = v[:, np.newaxis] >> np.arange(qm.shape[1])[np.newaxis, :] & 1    # ndarray[학생 id, 인지요소 id] -> 0 or 1 (가장 가능성 높은 인지요소 조합)

    return prediction
