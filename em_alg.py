import numpy as np
import pickle
from itertools import combinations
import time
from person_data import *
################################################################################
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return intersection / union
################################################################################
def extract_data(person_data):
    # Gather columnar data
    person_ids = np.array([ele[0] for ele in person_data])
    record_pairs = list(combinations(person_ids, 2)) # to be used later for pid lookup
    n = len(person_data[0]) - 1 # -1 for person id
    n_rows = len(record_pairs)
    n_cols = n + 2 # for g_m + g_u from Grannis et al.
    em_data = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        pid1, pid2 = record_pairs[i]
        data1 = person_data[np.where(person_ids == pid1)[0][0]]
        data2 = person_data[np.where(person_ids == pid2)[0][0]]
        
        # Features: gender, birth year, birth month, birth day, conditions, drugs
        for j in range(1, n+1):
            if type(data1[j]) == list:
                lst1 = [ele for ele in data1[j] if ele is not None]
                lst2 = [ele for ele in data2[j] if ele is not None]
                if len(lst1) > 0 and len(lst2) > 0:
                    sim_score = jaccard_similarity(data1[j], data2[j])
                    if sim_score > 0.1: # conditions
                        em_data[i][j-1] = 1
                    else:
                        em_data[i][j-1] = 0
            else:
                if data1[j] == data2[j]:
                    em_data[i][j-1] = 1
                else:
                    em_data[i][j-1] = 0
        # Randomly initialize gamma values
        em_data[i][-2] = 0.1*np.random.random()
        em_data[i][-1] = 0.01*np.random.random()

    # Additional preprocessing step to remove rows with all 0s?
    # em_data = em_data[~np.all(em_data[:,:-2] == 0, axis=1)]
    
    return em_data, record_pairs
################################################################################
def e_step(data, m, u, p, n, N):
    for i in range(N):
        gamma_j = data[i][:-2] # last 2 cols are g_m & g_u
        # gm determination
        m_term1 = p * np.prod([m[k]**(gamma_j[k])*(1-m[k])**(1-gamma_j[k]) 
                               for k in range(n)])
        m_term2 = m_term1 + (1-p)*np.prod([u[k]**(gamma_j[k])*(1-u[k])**(1-gamma_j[k]) 
                                           for k in range(n)])
        gm = m_term1 / m_term2
        #gu determination
        u_term1 = (1-p) * np.prod([u[k]**(gamma_j[k])*(1-u[k])**(1-gamma_j[k]) 
                                   for k in range(n)])
        u_term2 = u_term1 + p * np.prod([m[k]**(gamma_j[k])*(1-m[k])**(1-gamma_j[k]) 
                                         for k in range(n)])
        gu = u_term1 / u_term2
        # Update g values
        data[i][-2] = gm
        data[i][-1] = gu
    return data
################################################################################
def m_step(data, n, N): 
    # Parameter updates
    p = data[:,-2].sum() / N
    m = [(data[:,k] * data[:,-2]).sum() / data[:,-2].sum() for k in range(n)]
    u = [(data[:,k] * data[:,-1]).sum() / data[:,-1].sum() for k in range(n)]
    return m, u, p
################################################################################
def run_EM(data, num_iterations=10, p0=0.99, m0=0.9, u0=0.1):
    n = data.shape[1] - 2
    N = len(data)
    
    # Define the initial values for the model parameters
    m = np.ones(n) * m0
    u = np.ones(n) * u0
    p = p0
    
    # Run the EM algorithm
    for i in range(num_iterations):
        # time the iterations and print time for each iteration
        start = time.time()
        # E-step
        data = e_step(data, m, u, p, n, N)
        # M-step
        m, u, p = m_step(data, n, N)
        end = time.time()
        print(f"Iteration {i+1} took {end-start: .1f} seconds")
        
    print("m: ", m)
    print("u: ", u)
    print("p: ", p)
    
    return m, u, p, data
################################################################################
def score_pairs(m, u, p, result, record_pairs):
    N = len(result)
    n = result.shape[1] - 2
    scores = np.zeros(N)
    for i in range(N):
        gamma_j = result[i][:-2]
        score = sum(
            [np.log2((m[k]/u[k]))**(gamma_j[k]) * 
             np.log2((1-m[k])/(1-u[k]))**(1-gamma_j[k]) for k in range(n)]
        )
        scores[i] = score
    return scores
################################################################################
def fellegi_sunter_scores(person_data, num_iterations=10,p0=0.99,m0=0.9,u0=0.1):
    print("Extracting data...")

    main_start = time.time()
    data, record_pairs = extract_data(person_data)    
    end = time.time()
    
    print(f"Data extracted in {end-main_start:.1f} seconds," +
          f" {len(person_data)} records, {len(data)} pairs")
    
    # Run the EM algorithm
    m, u, p, result = run_EM(data, num_iterations)
    # Score the pairs
    scores = score_pairs(m, u, p, result, record_pairs)
    
    # Get the record pairs and scores in the right format
    pid1, pid2 = zip(*record_pairs)
    record_pair_scores = list(zip(pid1, pid2, scores))
    
    main_end = time.time()
    print(f"Total time: {main_end-main_start:.1f} seconds")
    
    return record_pair_scores, m, u, p
################################################################################
def get_top_scores(record_pair_scores):
    '''For each unique ID, returns the record pair with highest score.'''
    top_score = dict()
    for pid1, pid2, score in record_pair_scores:
        if top_score.get(pid1):
            if score > top_score[pid1][1]:
                top_score[pid1] = (pid2, score)
        else:
            top_score[pid1] = (pid2, score)


        if top_score.get(pid2):
            if score > top_score[pid2][1]:
                top_score[pid2] = (pid1, score)
        else:
            top_score[pid2] = (pid1, score)

    top_scores = np.array([(pid, other_pid, score) 
                       for pid, (other_pid, score) 
                       in top_score.items()])
    
    max_score = max(top_scores[:,2])
    top_top_scores = top_scores[top_scores[:,2] == max_score]
    detected_duplicates = [tuple(ele) if ele[0] > ele[1] 
               else tuple([ele[1], ele[0]]) 
               for ele in top_top_scores[:,0:2].astype(int)]
    detected_duplicates.sort(key=lambda x: x[0])
    return top_scores, detected_duplicates
################################################################################
if __name__ == "__main__":

    load_saved_data = True
    save = True
    n_iter = 1

    ############################################################################
    print("Getting data...")
    # Read in the data
    if load_saved_data:
        with open("./data/person_data.pkl", "rb") as f:
            person_data = pickle.load(f)
    else:
        person_data = get_data(gender_id=None, n=None)
    print("Data collected.")

    record_pair_scores, m, u, p = fellegi_sunter_scores(person_data, num_iterations=10)
    top_scores = get_top_scores(record_pair_scores)
    if save:
        print('Saving.')
        with open("./output/top_scores).p", "wb") as f:
            pickle.dump(top_scores, f)
        with open("./output/record_pair_scores).p", "wb") as f:
            pickle.dump(record_pair_scores, f)
################################################################################