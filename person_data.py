import numpy as np
import pickle
import psycopg2

def get_data(gender_id=None, n=None, save=False):
    con = psycopg2.connect(
        database="ohdsi",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432")
    cur = con.cursor()

    lim_statement = '' if n is None else f'LIMIT {n}'
    
    no_drugs = f'''
	SELECT person_id,
    gender_concept_id,
	year_of_birth,
	month_of_birth, 
	day_of_birth,
	array_agg(DISTINCT condition_occurrence.condition_concept_id) as conditions
	FROM cds_cdm.person
	LEFT JOIN cds_cdm.condition_occurrence USING (person_id)
	LEFT JOIN cds_cdm.drug_exposure USING (person_id)		  
	GROUP BY person_id, gender_concept_id, year_of_birth, month_of_birth, day_of_birth
    {lim_statement};'''

    gender_block = f'''
	SELECT person_id,
	year_of_birth,
	month_of_birth, 
	day_of_birth,
	array_agg(DISTINCT condition_occurrence.condition_concept_id) as conditions,
	array_agg(DISTINCT drug_exposure.drug_concept_id) as drugs
	FROM cds_cdm.person
	LEFT JOIN cds_cdm.condition_occurrence USING (person_id)
	LEFT JOIN cds_cdm.drug_exposure USING (person_id)
	WHERE gender_concept_id = {gender_id} OR gender_concept_id IS NULL			  
	GROUP BY person_id, year_of_birth, month_of_birth, day_of_birth
    {lim_statement};'''

    gender_block_no_drugs = f'''
	SELECT person_id,
	year_of_birth,
	month_of_birth, 
	day_of_birth,
	array_agg(DISTINCT condition_occurrence.condition_concept_id) as conditions
	FROM cds_cdm.person
	LEFT JOIN cds_cdm.condition_occurrence USING (person_id)
	WHERE gender_concept_id = {gender_id} OR gender_concept_id IS NULL			  
	GROUP BY person_id, year_of_birth, month_of_birth, day_of_birth
    {lim_statement};'''

    cur.execute(no_drugs)
    person_data = cur.fetchall()

    if save:
        # save person_data as pickle
        with open("./data/person_data.pkl", "wb") as f:
            pickle.dump(person_data, f)

    return person_data

if __name__ == "__main__":
    person_data = get_data(save=True)
