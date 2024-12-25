import copy
import numpy as np
class forward_eliminator:
    def __init__(self,array,scalers):
        self.array=array
        self.scalers=scalers
        self.tol=1e-9
        self.factors=[]
    def pivoting(self, row):
        index = row
        flag=False
        max_ratio = abs(self.array[row][row] / self.scalers[row])
        for i in range(row+1,len(self.array)):
            current_ratio = abs(self.array[i][row] / self.scalers[i])
            if current_ratio > max_ratio:
                max_ratio = current_ratio
                index = i
        if index != row:
            self.array[[row, index]] = self.array[[index, row]]
            self.scalers[row], self.scalers[index] = self.scalers[index], self.scalers[row]
            flag=True
        return self.array,flag
    def Forward_Elimination(self):
        arr = self.array    
        rows = len(arr)
        cols = len(arr[0])
        for k in range(rows):
            self.pivoting(k)
            if abs(arr[k][k] / self.scalers[k]) < self.tol:
                return -1
            for i in range(k + 1, rows):
                idx=0
                factor = arr[i][k] / arr[k][k]
                self.factors.append(factor)
                for j in range(k+1, cols):
                    arr[i][j] -= factor * arr[k][j]
                    
        if abs(arr[rows - 1][rows - 1] / self.scalers[rows - 1]) < self.tol:
            return -1
        self.array = arr
        return factor
    def Forward_Elimination_nextStep(self):
        flag = True 
        arr = self.array
        rows = len(arr)
        cols = len(arr[0])
        for i in range(rows):
            arr,bool=self.pivoting(i)
            if abs(arr[i][i] / self.scalers[i]) < self.tol:
                    return -1   
            if bool==True:
                return False,arr
            if any(arr[j][i] != 0 for j in range(i + 1, rows)):
                flag = False
                 
                for j in range(i + 1, rows):
               
                    if arr[j][i] != 0: 
                        factor = arr[j][i] / arr[i][i]
                        for k in range(i, cols):
                            arr[j][k] =arr[j][k]- (factor * arr[i][k])   
                        return flag,arr
        if abs(arr[rows - 1][rows - 1] / self.scalers[rows - 1]) < self.tol:
            return -1        
        return flag,arr


