"""
Convert party strings to integers
"""
import numpy as np

party_map_str = """
Centaur 1
Cosmos 2
Ebony 3
Odyssey 4
Tokugawa 5
"""

char_party = {}
char_party_float = {}
index_party = {}
for line in party_map_str.strip().split('\n'):
	ch, index = line.split()
	char_party_float[ch] = float(index)
	char_party[ch] = int(index)
	index_party[float(index)] = ch


target_class_str = """
Centaur 1.,0.,0.,0.,0
Cosmos 0.,1.,0.,0.,0
Ebony 0.,0.,1.,0.,0
Odyssey 0.,0.,0.,1.,0
Tokugawa 0.,0.,0.,0.,1
"""

char_target = {}
index_target = {}
for line in target_class_str.strip().split('\n'):
	ch, index = line.split(" ")
	char_target[ch] = [float(i) for i in index.strip().split(".,")]
#	index_target[index] = ch

"""
Convert age strings to integers
"""
age_map_str = """
18-24 1
25-35 2
36-45 3
46-55 4
55+ 5
"""

char_age = {}
index_age = {}
for line in age_map_str.strip().split('\n'):
	ch, index = line.split()
	char_age[ch] = float(index)
	index_age[float(index)] = ch

#print("{}",format(char_age['46-55']))
"""
Convert education degrees to integers
"""
education_map_str = """
Primary 1
Diploma 2
Degree 3
Masters 4
"""

char_education = {}
index_education = {}
for line in education_map_str.strip().split('\n'):
	ch, index = line.split()
	char_education[ch] = float(index)
	index_education[float(index)] = ch


"""
Convert ??
"""
