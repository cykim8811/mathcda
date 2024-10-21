
import numpy as np

class ACDM:
    def __init__(self, slip, guess):
        print("\033[93mWarning: Implementation of ACDM may be incorrect.\033[0m")
        # slip: ndarray[문항 id] -> 0~1 (인지요소를 모두 포함하는데도 틀릴 확률)
        # guess: ndarray[문항 id] -> 0~1 (인지요소를 모두 포함하지 않아도 맞출 확률)
        self.slip = slip
        self.guess = guess

    def P(self, alpha, qm):
        # alpha: ndarray[학생 id, 인지요소 id] -> 0 or 1 (해당 학생이 해당 인지요소를 가지고 있는지 여부)
        # qm: ndarray[문항 id, 인지요소 id] -> 0 or 1 (해당 문항이 해당 인지요소를 포함하는지 여부)

        # 형식을 [문항 id, 인지요소 id, 학생 id]로 맞춰주기
        alpha = alpha.T[np.newaxis, :, :]
        qm = qm[:, :, np.newaxis]

        d0 = self.guess[:, np.newaxis]
        d = ((1 - self.slip) / self.slip.shape[0])[:, np.newaxis, np.newaxis]

        return d0 + (alpha * d).sum(axis=1)  # ndarray[문항 id, 학생 id] -> 0~1 (해당 학생이 해당 문항을 맞출 확률)



    def L(self, alpha, qm, x):
        # x: ndarray[학생 id, 문항 id] -> 0 or 1 (해당 학생이 해당 문항을 맞췄는지 여부)
        
        p = self.P(alpha, qm)

        return (np.pow(p, x.T) * np.pow(1 - p, 1 - x.T)).prod(axis=0)  # ndarray[학생 id] -> 0~1 (Likelihood)

