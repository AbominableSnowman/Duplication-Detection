import random
import numpy as np
import pandas as pd
import pickle
from em_alg import *

TRUE_LINKS = [
        (7765, 2599),(7766, 2788),(7767, 3549),(7768, 2752),(7769, 2591),
        (7770, 3046),(7771, 3155),(7772, 3003),(7773, 2840),(7774, 3321),
        (7775, 3005),(7776, 3222),(7777, 3603),(7778, 3527),(7779, 3577),
        (7780, 2800),(7781, 2871),(7782, 2750),(7783, 3654),(7784, 3875)]
################################################################################
def sample_data(fraction_of_duplicates, person_data):
    '''Take a sample of data adjusting the total number according to a fixed
    number of duplicates and a desired fraction of duplicates.'''

    dup_pids = [item for sublist in TRUE_LINKS for item in sublist]
    other_pids = [ele[0] for ele in person_data if ele[0] not in dup_pids]
    N = len(person_data)
    sample_size = min(int(len(dup_pids) / fraction_of_duplicates), N) - len(dup_pids)
    selected_pids = dup_pids + random.sample(other_pids, sample_size)
    person_data_sample = [ele for ele in person_data if ele[0] in selected_pids]
    return person_data_sample
################################################################################
def get_performance(top_scores, detected_duplicates):
    N = len(top_scores)
    
    tp = 0
    fp = 0
    for ele in detected_duplicates:
        if ele in TRUE_LINKS:
            tp += 1
        else:
            fp += 1
    fn = (len(TRUE_LINKS) - tp)
    tn = (N - tp - fp - fn)

    #tp /= N
    #fp /= N
    #fn /= N
    #tn /= N

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    return tp, fp, tn, fn, specificity, sensitivity

################################################################################
def run_tests(load_saved_data=True, save=True):
    if load_saved_data:
        with open("./data/person_data.pkl", "rb") as f:
            person_data = pickle.load(f)
    ############################################################################
    # Initialize dataframe for results
    num_attr = len(person_data[0]) - 1 # -1 for pid
    results = pd.DataFrame(columns=['fraction_of_duplicates', 'num_iterations',
                                    'total_num_records', 'total_record_pairs',
                                    'detected_duplicates',
                                    'tp', 'fp', 'tn', 'fn', 'specificity', 
                                    'sensitivity', 'p_init', 'm_init', 'u_init', 
                                    'p'] + [f'm{i}' for i in range(num_attr)]
                                    + [f'u{i}' for i in range(num_attr)])
    ############################################################################
    # Results for different fractions of duplicates
    frac_dups = [0.01, 0.02, 0.05, 0.1, 0.12]
    num_iter = 15
    p0 = 0.5
    m0 = 0.9
    u0 = 0.1

    for fraction_of_duplicates in frac_dups:
        print(f'Data sample for fraction of duplicates: {fraction_of_duplicates:.2f}')
        person_data_sample = sample_data(fraction_of_duplicates, person_data)
        record_pair_scores, m, u, p = fellegi_sunter_scores(person_data_sample, 
                                                    num_iterations=num_iter,
                                                    p0=p0, m0=m0, u0=u0)
        top_scores, detected_duplicates = get_top_scores(record_pair_scores)
        tp, fp, tn, fn, specificity, sensitivity = get_performance(top_scores, 
                                                                   detected_duplicates)

        results_dict = {
            'fraction_of_duplicates': fraction_of_duplicates,
            'num_iterations': num_iter,
            'total_num_records': len(person_data_sample),
            'total_record_pairs': len(record_pair_scores),
            'detected_duplicates': len(detected_duplicates),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'p_init': p0, 'm_init': m0, 'u_init': u0, 'p': p}
        for i in range(num_attr):
            results_dict[f'm{i}'] = m[i]
            results_dict[f'u{i}'] = u[i]

        results = results.append(results_dict, ignore_index=True)

        if save:
            print('Saving.\n\n')
            with open(f"./output/frac_dup_test/record_pair_scores_" + \
                    f"{str(fraction_of_duplicates)}.p", "wb") as f:
                pickle.dump(record_pair_scores, f)
            with open(f"./output/frac_dup_test/top_scores_" + \
                    f"{str(fraction_of_duplicates)}.p", "wb") as f:
                pickle.dump(top_scores, f)
    
    ############################################################################
    # Results for different numbers of iterations
    num_iterations = np.arange(1, 16)
    p0 = 0.5
    m0 = 0.9
    u0 = 0.1
    fraction_of_duplicates = 0.08
    for num_iter in num_iterations:
        print(f'Data sample for number of iterations: {num_iter}')
        
        person_data_sample = sample_data(fraction_of_duplicates, person_data)
        record_pair_scores, m, u, p = fellegi_sunter_scores(person_data_sample, 
                                                    num_iterations=num_iter,
                                                    p0=p0, m0=m0, u0=u0)
        top_scores, detected_duplicates = get_top_scores(record_pair_scores)
        tp, fp, tn, fn, specificity, sensitivity = get_performance(top_scores, 
                                                                   detected_duplicates)

        results_dict = {
            'fraction_of_duplicates': fraction_of_duplicates,
            'num_iterations': num_iter,
            'total_num_records': len(person_data_sample),
            'total_record_pairs': len(record_pair_scores),
            'detected_duplicates': len(detected_duplicates),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'p_init': p0, 'm_init': m0, 'u_init': u0, 'p': p}
        for i in range(num_attr):
            results_dict[f'm{i}'] = m[i]
            results_dict[f'u{i}'] = u[i]

        results = results.append(results_dict, ignore_index=True)

        if save:
            print('Saving.\n\n')
            with open(f"./output/iteration_test/rps_iters_"+\
                      f"{str(num_iter)}.p", "wb") as f:
                pickle.dump(record_pair_scores, f)
            with open(f"./output/iteration_test/ts_iters_"+\
                      f"{str(num_iter)}.p", "wb") as f:
                pickle.dump(top_scores, f)
    ############################################################################
    # Save results
    if save:
        with open("./output/all_test_results.pkl", "wb") as f:
            pickle.dump(results, f)
    return results