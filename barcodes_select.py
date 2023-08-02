import numpy as np
import pandas as pd
import tqdm

global atcgtoarray 

atcgtoarray = {'A': np.array([[1,0,0,0]]), 'T': np.array([[0,1,0,0]]), 'C': np.array([[0,0,1,0]]), 'G': np.array([[0,0,0,1]])}

def hamming_distance(chaine1, chaine2):
    
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))


def atcgtoarray_func7(barcodes):
    
    return np.concatenate((atcgtoarray[barcodes[0]], atcgtoarray[barcodes[1]], atcgtoarray[barcodes[2]],
                           atcgtoarray[barcodes[3]], atcgtoarray[barcodes[4]], atcgtoarray[barcodes[5]],
                           atcgtoarray[barcodes[6]]), axis=0)

def atcgtoarray_func8(barcodes):
    
    
    return np.concatenate((atcgtoarray[barcodes[0]], atcgtoarray[barcodes[1]], atcgtoarray[barcodes[2]],
                           atcgtoarray[barcodes[3]], atcgtoarray[barcodes[4]], atcgtoarray[barcodes[5]],
                           atcgtoarray[barcodes[6]], atcgtoarray[barcodes[7]]), axis=0)

def atcgtoarray_func9(barcodes):
    
    return np.concatenate((atcgtoarray[barcodes[0]], atcgtoarray[barcodes[1]], atcgtoarray[barcodes[2]],
                           atcgtoarray[barcodes[3]], atcgtoarray[barcodes[4]], atcgtoarray[barcodes[5]],
                           atcgtoarray[barcodes[6]], atcgtoarray[barcodes[7]], atcgtoarray[barcodes[8]]), axis=0)

def atcgtoarray_func10(barcodes):
    
    return np.concatenate((atcgtoarray[barcodes[0]], atcgtoarray[barcodes[1]], atcgtoarray[barcodes[2]],
                           atcgtoarray[barcodes[3]], atcgtoarray[barcodes[4]], atcgtoarray[barcodes[5]],
                           atcgtoarray[barcodes[6]], atcgtoarray[barcodes[7]], atcgtoarray[barcodes[8]], 
                           atcgtoarray[barcodes[9]]), axis=0)

def atcgtoarray_func11(barcodes):
    
    return np.concatenate((atcgtoarray[barcodes[0]], atcgtoarray[barcodes[1]], atcgtoarray[barcodes[2]],
                           atcgtoarray[barcodes[3]], atcgtoarray[barcodes[4]], atcgtoarray[barcodes[5]],
                           atcgtoarray[barcodes[6]], atcgtoarray[barcodes[7]], atcgtoarray[barcodes[8]], 
                           atcgtoarray[barcodes[9]], atcgtoarray[barcodes[10]]), axis=0)

atcgtoarray_func = {7: atcgtoarray_func7, 8: atcgtoarray_func8, 9: atcgtoarray_func9, 10: atcgtoarray_func10, 11: atcgtoarray_func11}

def distance_func(barcode1, barcode2, accurancy, n):

    return np.prod((atcgtoarray_func[n](barcode1).dot(accurancy) *  atcgtoarray_func[n](barcode2)).sum(-1))

def distance_mismatch(barcodes_list, n, total_barcodes):
    
    distance_mis = n*np.ones((len(barcodes_list), len(barcodes_list)), dtype=float)
    for i in range(total_barcodes):
        for j in range(total_barcodes):
            if j != i :
                distance_mis[i,j] = hamming_distance(barcodes_list[i], barcodes_list[j])

   # distance_mis[distance_mis < boundary] = 100
   # distance_mis[distance_mis > boundary] = 0
    
    return distance_mis

def distance_delete(barcodes_list, n, total_barcodes):
    
    distance_delete = (n-1)*np.ones((total_barcodes, total_barcodes), dtype=float)

    for i in tqdm.tqdm(range(len(barcodes_list))):
        for j in range(len(barcodes_list)):
            if i != j:
                distance2 = (n-1)*np.ones((n, n), dtype=float)
                for ii in range(n):
                    for jj in range(n):
                        distance2[ii,jj] = hamming_distance(barcodes_list[i][:ii]+barcodes_list[i][ii+1:], barcodes_list[j][:jj]+barcodes_list[j][jj+1:])

                distance_delete[i,j] = distance2.min()

   # distance_delete[distance_delete < boundary] = 100
   # distance_delete[distance_delete > boundary] = 0
        
    return distance_delete


def distance_insertion(barcodes_list, n, total_barcodes):
    
    ATCG = {0:'A', 1:'T', 2: 'C', 3: 'G'}
    distance333 = (n+1)*np.ones((4, 4), dtype=float) 
    distance_insert = (n+1)*np.ones((total_barcodes, total_barcodes), dtype=float)
    
    for i in tqdm.tqdm(range(total_barcodes)):
        for j in range(total_barcodes):
            if i != j:
                
                distance33 = (n+1)*np.ones((n, n), dtype=float)
                
                for ii in range(n):
                    for jj in range(n):

                        for iii in range(4):
                            for jjj in range(4):

                                distance333[iii,jjj] = hamming_distance(barcodes_list[i][:ii]+ATCG[iii]+barcodes_list[i][ii:], 
                                                                        barcodes_list[j][:jj]+ATCG[jjj]+barcodes_list[j][jj:])


                        distance33[ii,jj] = distance333.min()

                for iii in range(4):
                    for jjj in range(4):

                        distance33[ii,jj] = min(hamming_distance(barcodes_list[i]+ATCG[iii], barcodes_list[j]+ATCG[jjj]), distance33[ii,jj] )


                distance_insert[i,j] = distance33.min()
                
   # distance_insert[distance_insert < boundary] = 100
   # distance_insert[distance_insert > boundary] = 0
        
    return distance_insert

def barcode_num_func(barcode_list,barcode_num):
    
    if len(barcode_list) >= barcode_num:
        return barcode_num
    else:
        raise Exception("Sorry, no enough candidate barcodes.")
        

class Barcodes:
    
    def __init__(self, barcode_list, barcode_num = 50, tolerance = 1, accurancy_matrix = None):
        
        self.barcode_list = barcode_list
        self.barcode_num = barcode_num_func(self.barcode_list,barcode_num)
        self.n = len(barcode_list[0])
        self.accurancy_matrix = accurancy_matrix
        self.total_barcodes = len(barcode_list)
        self.boundary_mismatch = 2*tolerance+1
        self.boundary_delete = tolerance
        self.boundary_insert = tolerance
        
        
        self.distance_mismatch_matrix = distance_mismatch(self.barcode_list, self.n, self.total_barcodes)
        self.distance_delete = distance_mismatch(self.barcode_list, self.n, self.total_barcodes)
        self.distance_insertion = distance_insertion(self.barcode_list, self.n, self.total_barcodes)
        
    def basic_barcodes(self):
        
        mismatch_temp = self.distance_mismatch_matrix.copy()
        mismatch_temp[mismatch_temp < self.boundary_mismatch] = -self.n*10
        
        delete_temp = self.distance_delete.copy()
        delete_temp[delete_temp < self.boundary_delete] = -self.n*10
        
        insertion_temp = self.distance_insertion
        insertion_temp[insertion_temp < self.boundary_insert] = -self.n*10
        
        new = mismatch_temp + delete_temp + insertion_temp
        index = list(range(self.total_barcodes))
        for i in range(self.total_barcodes - self.barcode_num):
            minarg = np.where(new == np.nanmin(new))[0][0]
            index.remove(minarg)
            new[minarg,:] = np.nan
            new[:,minarg] = np.nan
            
        final = new[np.ix_(index,index)]

        if final.min() < 0:
            print('No valid barcodes')
        else:
            self.barcodes = np.array(self.barcode_list)[index]
            return np.array(self.barcode_list)[index]
        
    def calculate_probability(self):
    
        if type(self.accurancy_matrix) == np.ndarray:

            Prob = self.n*np.ones((self.total_barcodes, self.total_barcodes), dtype=float)
            for i in range(self.total_barcodes):
                for j in range(self.total_barcodes):

                     Prob[i,j] = distance_func(self.barcode_list[i], self.barcode_list[j], self.accurancy_matrix, self.n)
                        
            self.prob_matirx = Prob
            
            return Prob

        else:
            print("No accurancy matirx input!")
            
    def prob_barcodes(self):

        new = self.calculate_probability().copy()
        np.fill_diagonal(new,0)

        new[self.boundary_mismatch < self.boundary_mismatch] = self.n*10
        new[self.boundary_delete < self.boundary_delete] = self.n*10
        new[self.boundary_insert < self.boundary_insert] = self.n*10
        
        index = list(range(self.total_barcodes))
        for i in range(self.total_barcodes - self.barcode_num):
            maxarg = np.where(new == np.nanmax(new))[0][0]
            index.remove(maxarg)
            new[maxarg,:] = np.nan
            new[:,maxarg] = np.nan
            
        final = new[np.ix_(index,index)]

        if final.max() > 1:
            print('No valid barcodes')
        else:
            self.final_prob_matirx = final
            self.final_barcodes = np.array(self.barcode_list)[index]
            
            return np.array(self.barcode_list)[index]
    