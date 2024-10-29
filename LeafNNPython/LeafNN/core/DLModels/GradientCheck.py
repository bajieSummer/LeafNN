import copy
import numpy as np
from LeafNN.utils.MathUtils import MathUtils
from LeafNN.utils.Log import Log

class GradientCheck:
    """
    iterationNum: current iteration number
    checkFrequency: the frequency of gradient checking
    """
    def Check(iterationNum,checkFrequency,wb,grads,calCostWithParams,*args_dataxy):
        Log.Debug("check_gradients",f"check_gradients begin>>iterationNum={iterationNum}")
        if(iterationNum%checkFrequency !=0):
            return True
        # wn1 = w_ij(l)+ ep1 
        # wn2 = w_ij(l)- ep1
        w = wb[0]
        b = wb[1]
        layers = len(w)
        cacheWn1 = copy.deepcopy(w)
        cacheCheckDlDW = [None]*(layers) 
        # bn1 = b_j(l)+ ep1 
        # bn2 = b_j(l)- ep1
        cachebn1 = copy.deepcopy(b)
        cacheCheckDlDB = [None]*(layers)
        isCheckPass = True 
        for l in range(layers):
            nl1,nl2 = w[l].shape
            cacheCheckDlDW[l] = np.zeros([nl1,nl2])
            cacheCheckDlDB[l] = np.zeros([1,nl2])
            Log.Debug("Gradient_Check",f"checkgradients_in layer{l},l nodes={nl1}, l+1 nodes={nl2},dldB shape={grads[0][l].shape}")
            for i in range(nl1):
                for j in range(nl2):
                    eps = MathUtils.Epsilon
                    cacheWn1[l][i][j] += eps
                    J1 = calCostWithParams([cacheWn1,b],*args_dataxy)
                    cacheWn1[l][i][j] -= 2*eps
                    J2 = calCostWithParams([cacheWn1,b],*args_dataxy)
                    chekdW = J1/(2.0*eps) - J2/(2.0*eps)
                    origindW = grads[0][l][i][j]
                    Log.Debug("Gradient_Check",f"checkgradients_dldw l={l} i={i},j={j} cost1>{J1} cost2>{J2},checkdldw={chekdW},computerdldw = {origindW}")
                    cacheCheckDlDW[l][i][j]= chekdW 
                    if(np.abs(origindW - chekdW)>MathUtils.GradCheckMinDiff):
                        Log.Error("Gradient_Check",f"gradient checking: computing_dldw_wrong : checkDiff chekdW={chekdW},computerdldw = {origindW},layer={l},i={i},j={j}:")
                        isCheckPass = False
                    #recover wij
                    cacheWn1[l][i][j] += eps
                    #computer b
                    if( i == 0):
                        cachebn1[l][i][j] += eps
                        # outputY1,cacheA,cacheZ = self.predictWithParams(self.trainX,self.modelWeights,cachebn1)
                        # J1 = self.__calCost(outputY1)
                        J1 = calCostWithParams([w,cachebn1],*args_dataxy)
                        cachebn1[l][i][j] -= 2*eps
                        # outputY2,cacheA,cacheZ = self.predictWithParams(self.trainX,self.modelWeights,cachebn1)
                        # J2 = self.__calCost(outputY2)
                        J2 = calCostWithParams([w,cachebn1],*args_dataxy)
                        checkDB =  (J1-J2)/(2*eps)
                        originDB = grads[1][l][i][j]
                        cachebn1[l][i][j] += eps
                        cacheCheckDlDB[l][i][j] = checkDB
                        Log.Debug("Gradient_Check",f"checkgradients_dldb l={l},i={i},j={j} dldb={originDB} checkDB={checkDB} j1={J1},j2={J2}")
                        if(np.abs(originDB - checkDB)>MathUtils.GradCheckMinDiff):
                            Log.Error("Gradient_Check",f"gradient checking: computing_dldb_wrong : checkDiff diffDB={checkDB}: computer dldb:{originDB} layer={l},i={i},j={j} diff={originDB - checkDB}")
                            isCheckPass = False
        if not isCheckPass:
           Log.Error("Gradient_Check",f"Check Not Passed: the differences between check result and gradient computing:")
        return isCheckPass