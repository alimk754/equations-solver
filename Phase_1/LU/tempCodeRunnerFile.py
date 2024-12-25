 if abs(arr[k][k] / self.scalers[k]) < self.tol:
                yield {'matrix': arr.copy()}
                return