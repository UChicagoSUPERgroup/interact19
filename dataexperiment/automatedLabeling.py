import json
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
import re

import os
import subprocess as sp
import argparse

import itertools as it
from functools import reduce

import sympy
from sympy.logic import simplify_logic

parser = argparse.ArgumentParser()
parser.add_argument("concepts", 
    help='1. "test": find the best predicting rules for concepts \'nightstand\' and \'bride\'; '+
         '2. "all": find the best predicting rules for all labels in the Visual Genome dataset who are in at least [CONCEPT_CUTOFF] images; '+
        '3. "userstudy-auto": find the best predicting rules for \'nightstand\', \'old\', \'eat\', and \'intersection\' in the Visual Genome dataset; '+
        '4. "userstudy-manual": similar to "userstudy-auto", but form the candidate rules using the words that user study participants most commonly used to explain whether a concept is in an image.', 
        type=str, choices=["test", "all", "userstudy-auto", "userstudy-manual"])
args = parser.parse_args()

##################################################
########### Some global variables ################
##################################################
# For every [AUTOSAVE] target concepts, this code writes their results to a new csv file.
# The pool of target concepts consist of any label that has been assigned to 
# at least [CONCEPT_CUTOFF] images. The paper uses 100 for both.
AUTOSAVE = 100
CONCEPT_CUTOFF = 100
# Only labels that have been assigned to at least [CUTOFF] images are used for generating
# potential definitions of a given target concept. The paper uses 10.
LABEL_CUTOFF = 10 
# When analyzing F1 scores, only definitions that successfully predicts the existence of 
# a target concept in at least [TRUE_CUTOFF] images are considered. The paper uses 10.
TRUE_CUTOFF = 10 
# Display the F1 scores with [DISP] precision.
DISP = "{0:.4f}" 
# Save any files to [RESULTS_DIR].
RESULTS_DIR = 'results/' 
print("CUTOFF for labels: "+str(LABEL_CUTOFF))
print("CUTOFF for correct predictions: "+str(TRUE_CUTOFF))
print("CUTOFF for concepts: "+str(CONCEPT_CUTOFF))

def getAliases(alias_file):
    ''' Opens a object/relationship/attribute_alias.txt file. In the file, each line 
    contains synonyms for a given label (ex. "bike" and "a bike" are synonyms in this case). 
    Converts its contents to a dictionary in which:
    key = the first word/phrase of each line (and treated as a label that we would consider later),
    value = the rest of the synonyms on the same line as a list. '''
    with open(alias_file) as f:
        aliases = {}
        for line in f:
            alias = line.rstrip('\n').split(",") # list
            if (len(alias) > 1): # else ignore -- no alias for this word/phrase
                for a in alias[1:]:
                    aliases[a] = alias[0] 
    return aliases

def getImgID(img): # img = dict
    ''' Return the id (int) of an image. '''
    if type(img) == type({}): # for rel_img, which is a dict representation of an img
        return int(img['image_id'])
    elif type(img) == type(1): # index of the image in the dataframe
        return int(ATTRIB_DATA['image_id'][img])

##################################################
################### Load data ####################
##################################################
DATASET_DIR = "visualgenome-1.4/"
obj_alias_file = DATASET_DIR+"object_alias.txt"
OBJ_ALIASES = getAliases(obj_alias_file)

rel_alias_file = DATASET_DIR+"relationship_alias.txt"
REL_ALIASES = getAliases(obj_alias_file)

obj_synsets_file = DATASET_DIR+"object_synsets.json"
with open(obj_synsets_file) as f:
    OBJ_SYNSETS = json.load(f)

attr_synsets_file = DATASET_DIR+"attribute_synsets.json"
with open(attr_synsets_file) as f:
    ATTR_SYNSETS = json.load(f)

rel_synsets_file = DATASET_DIR+"relationship_synsets.json"
with open(rel_synsets_file) as f:
    REL_SYNSETS = json.load(f)

attrib_filename = DATASET_DIR+"attributes.json"
ATTRIB_DATA = pd.read_json(attrib_filename)

OBJS_BY_IMG = ATTRIB_DATA['attributes']

REL_FILE = DATASET_DIR+"extracted_relationships.txt"

def getRels():
    rel_data_lst = []
    try:
        with open(REL_FILE) as f:
            rel_data_lst = parseRels(f)
    except FileNotFoundError:
        print("%s was not found. Generating one now..." % REL_FILE)
        cmd_extract_rels = "ag -o \'(predicate\":[^,]+,)|(relationship_id\":[^,]+, \"synsets\":[^,]+,)|(image_id\": [^}]+)\' %srelationships.json > %s" %(DATASET_DIR, REL_FILE)
        # For our purpose this works, but it's a potential security hazard if the cmd gets changed
        sp.call(cmd_extract_rels, shell=True) 
        print("%s generated." % REL_FILE)
        with open(REL_FILE) as f:
            rel_data_lst = parseRels(f)
    return rel_data_lst

def parseRels(f):
    ''' Opens visualgenome/extracted_relationships.txt and converts its content into
    a list of dictionaries, one per image. Each dictionary is of the form
    {'image_id': int, 
    'rel':[[predicate1, synset1], [predicate2, synset2], ...]}'''
    rel_data_lst = []

    rel_dict_for_img = {}
    rels_for_img = [] # list of [predicate, synset]
    rel_pair = [] # to be a pair of [predicate, synset] eventually
    for line in f:
        parsed_line = [x.split(": ") for x in \
            "".join(line.rstrip('\n').rstrip(",").split("\"")).split(", ")] # list
        starter = parsed_line[0][0]
        if starter=='predicate': # ex. [['predicate', 'next to']]
            pred = parsed_line[0][1]
            rel_pair.append(pred)
        elif starter=='relationship_id': # [['relationship_id', '3186270'], ['synsets', '[by.r.01]']]
            synset_str = parsed_line[1][1]
            synset = '' if synset_str == '[]' else synset_str.lstrip('[').split('.')[0]
            rel_pair.append(synset)
            rels_for_img.append(rel_pair)
            rel_pair = [] # done with it -> reset
        elif parsed_line[0][0]=='image_id':
            rel_dict_for_img['image_id'] = int(parsed_line[0][1]) # not sure if necessary to convert to int
            rel_dict_for_img['rel'] = rels_for_img
            rel_data_lst.append(rel_dict_for_img)
            rels_for_img = [] # reset for the next img
            rel_dict_for_img = {} # reset for the next img
    return rel_data_lst

REL_DATA_LST = getRels()

##################################################
################### Parse data ###################
##################################################
nltk.download('wordnet')
LEM = WordNetLemmatizer()

def getWord(s): # maybe... we should just strip _, -, ', and anything except for letters
    '''Given a string s, remove any punctuations (except for apostrophe, to
    distinguish labels like "boy" vs. "boy's"), merge it into one word
    if it's a phrase, and return the result in lowercase. '''
    new_s = s.lower()
    res = re.findall(r"[a-z']+", new_s)
    return "".join(" ".join(res).split("_"))

def parseSynset(s):
    ''' Given a synset string, get the word part of the synset. '''
    return "".join(s.split(".")[0].lower().split("_"))

def convertToSynset(s, d): # d = synsets dict
    ''' After preprocessing the string s (see getWord()), 
    check if it has a synset using the synsets dictionary d. 
    Return the word part of the synset if it does, else
    check try to find its aliases/synonyms. '''
    new_s = getWord(s)
    if new_s in d:
        return parseSynset(d[new_s])
    else:
        # look for it in aliases
        new_s = OBJ_ALIASES.get(new_s, new_s) 
        # try again
        if new_s in d:
            return parseSynset(d[new_s])
        # just return the original, but as one word (ex. "night stand" -> "nightstand")
        return moreParsing(new_s)

def moreParsing(s):
    ''' Uses lemming along with some other filters, after a string has gone through 
    getWord(), synset lookup, and alias lookup, and did not find a more standardized match. '''
    return LEM.lemmatize("".join("".join(s.split(" ")).split("_"))) 

def getObjNameInAttr(obj): # returns list
    """ Given a dict representing an object label, return its name(s) in a list.
    Get nouns from 'synsets' or, if synsets not available, 'names'"""
    if len(obj['synsets']) != 0: # default to synsets
        return [parseSynset(item) for item in obj['synsets']]
    else:
        return [convertToSynset(n, OBJ_SYNSETS) for n in obj['names']] 

def getAttribsList(obj):
    """ Given a dict representing an object label, return its attributes as a list of strings.
    If it has no attributes, return an empty list. """
    return [convertToSynset(o, ATTR_SYNSETS) for o in obj.get('attributes', [])]

def getObjAttribLabels(img): # img = list of dicts
    ''' Given an img (a row of a dataframe containing objects/attributes of all images in the
    dataset), get the object and attribute labels assigned to it.'''
    obj_labels = set() 
    attr_labels = set()
    for obj in img: # obj = dict
        # get the real obj_name if there's one (from aliases), 
        # otherwise settle for the object label
        for real_obj_name in getObjNameInAttr(obj):
            if real_obj_name != '':
                obj_labels.add(real_obj_name)
        for real_attrib_name in getAttribsList(obj): 
            if real_attrib_name != '':
                attr_labels.add(real_attrib_name)
    return (obj_labels, attr_labels)

def getRelLabels(img): # img = list of dicts
    ''' Given an img (a row of a dataframe containing relationships of all images in the dataset),
    get the relationship labels assigned to it. '''
    rel_labels = set()
    for rel_pair in img['rel']:
        if rel_pair[1]!='': # has synset
            synset = rel_pair[1]
            rel_labels.add(synset)
        else: # doesn't have synset
            pred = getWord(rel_pair[0]) # pred
            synset = pred # init
            if pred in REL_SYNSETS:
                synset = REL_SYNSETS[pred].split(".")[0]
            else:
                pred = REL_ALIASES.get(pred, pred)
                synset = moreParsing(pred)
            if synset != '':
                rel_labels.add(synset)
    return rel_labels

def getImgLabels(img, rel_img):
    ''' Given an img (dict), return the set of object, attribute, and relationship labels 
    that appear in it. '''
    (obj_labels, attr_labels) = getObjAttribLabels(img)
    rel_labels = getRelLabels(rel_img)
    return (obj_labels, attr_labels, rel_labels)

##################################################
################# Process data ###################
##################################################
def calcF1(precision, recall):
    ''' Given precision and recall, return the F1 score. '''
    f1 = 2 * ((precision*recall)/(precision+recall)) \
        if (precision+recall)!=0 else float(0)
    return f1

def incLabelType(labels, label_type_tracker, l_type):
    ''' Given labels (list of strs) of l_type (str) belonging to an image, increment count of 
    that type for each label in label_type_tracker (dict). This modifies label_type_tracker.'''
    for l in labels:
        if l not in label_type_tracker:
            label_type_tracker[l] = {'obj':0,'attr':0,'rel':0}
        label_type_tracker[l][l_type]+=1

def checkRelImgConsistencyWithObjAttrib(img_i, rel_img):
    ''' Given a representation for an image's objects/attributes (its index, to be specific,
    because its representation is a list that doesn't contain image_id info), 
    and a representation for an image's relationships, check that the two refer to the same image.'''
    if getImgID(img_i) != getImgID(rel_img):
        print("Uh-oh, image_id mismatch between obj/attr and rel for:",getImgID(img_i), 'and', getImgID(rel_img))

def countAllCoLabels(images, rel_imgs, img_start_i=0, img_end_i=OBJS_BY_IMG.size-1):
    ''' Go through all the images and do the following:
    1. Tally the number of images in which each label appear (img_counts);
    2. Determine the type of each label (object, attribute, or relationship) by getting
    the type that it most frequently appeared as (label_type_tracker);
    3. Count the number of images in which each pair of labels appear ("co-occurrence" of two labels)
    (all_label_counts).'''
    # key = label, value = dictionaries (whoe key = another label, 
    # value = number of images to which the two labels have been assigned)
    all_label_counts = {} 
    # key = label, value = number of images it was assigned to
    img_counts = {} 
    # key = label, value = dict that tracks tally of the label being of type 'obj', 'attr', or 'rel'
    label_type_tracker = {} 
    
    for i in range(img_start_i, img_end_i+1):
        img = images[i] # img = list of dicts, images = dataframe of lists
        rel_img = rel_imgs[i]
        checkRelImgConsistencyWithObjAttrib(i, rel_img)
            
        (obj_labels, attr_labels, rel_labels) = getImgLabels(img, rel_img)
        
        incLabelType(obj_labels, label_type_tracker, 'obj')
        incLabelType(attr_labels, label_type_tracker, 'attr')
        incLabelType(rel_labels, label_type_tracker, 'rel')
        labels_in_img = obj_labels | attr_labels | rel_labels 
                    
        for label in labels_in_img:
            # Increment count of images for each label in the image.
            img_counts[label] = img_counts.get(label,0)+1 
            
            # Increment count of images for every pair of labels that appear in this image.
            # The type of label (object, attribute, relationship) are not distinguished.
            if label not in all_label_counts:
                all_label_counts[label] = {}
            for colabel in labels_in_img:
                if label != colabel:
                    all_label_counts[label][colabel] = all_label_counts[label].get(colabel,0)+1

    label_types = {}
    for l in label_type_tracker:
        label_types[l] = max(label_type_tracker[l], key = label_type_tracker[l].get)
    return (all_label_counts, img_counts, label_types)

def getMaxCoLabels(concept): 
    """ Given a concept (which is a label), sort the list of co-occurring labels ("colabels") 
    for this concept by decreasing order of F1 score when using the co-occurring label as a 
    rule to predict concept's presence in an image"""
    
    max_colabels_cols = ["Co-Label for Concept \'"+concept+"\'", 'F1 Score', 'Precision','Recall', \
                        'True Pos (tp, # imgs with concept and rule)', \
                        'All Pos (tp+fp, # imgs for this rule total)', \
                        'All Instances of Concept (tp+fn, # imgs for concept total)']
                        
    # skip concepts that only appear in less than [CONCEPT_CUTOFF] images
    if IMG_COUNTS[concept] < CONCEPT_CUTOFF:
        return ([],[])
    
    # sort colabels by decreasing frequency of co-occurring labels
    sorted_colabels_freq = sorted(ALL_LABEL_COUNTS[concept], 
                            key=ALL_LABEL_COUNTS[concept].get, reverse=True) 
    
    lst = [] # just for information: printing out colabels by decreasing co-occ freq
    rows = [] # for the output dataframe
    rules = [x for x in sorted_colabels_freq if IMG_COUNTS[x]>=LABEL_CUTOFF]
    for rule in rules: # rule = colabel
        freq = ALL_LABEL_COUNTS[concept][rule] # tp
        pos = IMG_COUNTS[rule] # tp+fp
        targs = IMG_COUNTS[concept] # tp+fn

        (precision, recall) = calcPR(freq, pos, targs)
        f1_score = DISP.format(calcF1(precision, recall))
        (precision, recall) = (DISP.format(precision), DISP.format(recall))
        
        lst.append((rule, freq, precision, recall, f1_score)) 
        d_vals = [rule, f1_score, precision, recall, freq, pos, targs]
        addDfRow(rows, max_colabels_cols, d_vals)
        
    lst = sorted(lst, key=lambda x: x[4], reverse=True) # x[4] = f1_score
    rows = sorted(rows, key=lambda x: x['F1 Score'], reverse=True)
    df_name = RESULTS_DIR+concept+'_colabels_F1-sort.csv'
        
    df = pd.DataFrame(rows, columns=max_colabels_cols)
    
    df.to_csv(df_name)
    return (lst, df)

# modified from: https://docs.python.org/3/library/itertools.html#recipes
def genPowerset(iterable):
    ''' Generates a dictionary where key = length of a set, value = list of powersets of the 
    iterable of that length.'''
    s = list(iterable)
    powerset = list(it.chain.from_iterable(it.combinations(s, r) for r in range(1, len(s)+1)))
    result = {} # keys = length of elts
    for p in powerset: # always sort the tuples for easier search later
        result[len(p)] = result.get(len(p), [])+[tuple(sorted(p))]
    return result

def getCandidateRules(target_labels_list):
    ''' Given a list of groups (each group being a smaller list containing several target labels), 
    for each group:
    1. Identify all possible conjunctions that can be constructed from this group of target labels, 
    which is just the powerset of the group;
    2. Generate a hierarchy (dict) in which the smaller conjunctions are grouped under their supersets.
    Note that the groups are processed independently. Each conjunction is represented as a tuple
    of its terms (labels). This superset organization is useful for later: if a bigger conjunction holds
    true for some image, then a subset of that conjunction is automatically true and does not need to be
    checked.'''

    unique_target_labels = set()
    
    # for each group of target_labels in target_labels_list:
    # 1. form a powerset (all possible conjunctions, including singletons) of the 
    # members of this target_labels
    # 2. add powerset to target_combs_dict for a later fn
    # 3. find all unique_target_labels
    target_combs_dict = {} 
    for target_labels in target_labels_list:
        powersets = genPowerset(target_labels) #1
        target_combs_dict[target_labels[0]]=powersets #2
        
        for label in target_labels: #3
            unique_target_labels.add(label)
    return (target_combs_dict, list(unique_target_labels))

def countTargetLabels(total_imgs, target_labels_list):
    ''' Given a list (target_labels_list) of several lists, each smaller list containing potential 
    terms (labels) to form a conjunction rule, count the number of images that satisfy this rule.

    Args:
    total_imgs = number of images we want to analyze; target_labels_list = list of target_labels.
    Each target_labels is a list with the target_concept at the beginning, followed by labels that
    may be used to form a conjunctive rule (e.g. A & B) that predicts that target_concept. For each image, 
    we count the biggest conjunction (the combination of the most labels in target_labels).'''

    (target_combs_dict, unique_target_labels_lst) = getCandidateRules(target_labels_list)

    # List of dictionaries that keep track of number of images with a certain conjunction based on the labels
    and_counts_dict = {}
    for lst in target_labels_list:
        target_concept = lst[0]
        and_counts_dict[target_concept] = {}
        
    # For each img, increment count of imgs in which a conjunction (the biggest possible one from the corresponding
    # smaller list of target_labels) held true
    for i in range(0, total_imgs):
        img = OBJS_BY_IMG[i] 
        rel_img = REL_DATA_LST[i]
        checkRelImgConsistencyWithObjAttrib(i, rel_img)
        
        (obj_labels, attr_labels, rel_labels) = getImgLabels(img, rel_img)
        labels_in_img = obj_labels | attr_labels | rel_labels

        for target_labels in target_labels_list:
            target_concept = target_labels[0]
            # powerset (all possible conjunctions, including singletons) 
            # of the members of the corresponding target_labels
            target_combs = target_combs_dict[target_concept] # target_combs is also a dict            
            and_counts = and_counts_dict[target_concept]

            # from big to small combs: max, max-1, ... 2, 1 
            # elts in target_combs_dict correspond to elts in target_labels_list
            for l in range(max(target_combs.keys()), 0, -1): 
                # there should be AT MOST ONE comb (the biggest one) that
                # can be found in labelsInImg, because otherwise a larger comb
                # would've been chosen and the loop would've stopped
                # combFound: marks whether a conjunction (also the biggest possible one) has been found 
                # in this img's labels
                combFound = False 
                for comb in target_combs[l]:
                    if comb not in and_counts:
                        and_counts[comb] = 0
                        
                    # The following can only occur a max of once for a specific size of comb,
                    # because otherwise it would've happened to a larger size and then broke
                    if set(comb).issubset(labels_in_img): 
                        and_counts[comb] += 1
                        if combFound == True:
                            print(">>> UH-OH, a conjunction has already been found?")
                            return () # which results in an error in the next fn
                        combFound = True
                        break # stop checking the rest of possible conjunctions for length l
                if combFound: # stop checking smaller conjunctions, since a bigger conjunction has what we need
                    break

    return (and_counts_dict, target_combs_dict)

def calcPR(rule_true, rule_total, concept_total):
    """Given the number of true positives (rule_true), all positives (rule_total), 
    and all instances with the concept present (concept_total), calculate the precision and recall."""
    precision = float(rule_true/rule_total) if rule_total != 0 else float(0) # tp/(tp+fp)    
    recall = float(rule_true/concept_total) # tp/(tp+fn)
    return (precision, recall)

def addDfRow(rows_list, row_keys, row_vals):
    ''' Update rows_list with a row, which is a dictionary with row_keys and row_vals.
    Assume that all 3 args are lists.'''
    row = {k : v for (k, v) in zip(row_keys, row_vals)}
    rows_list.append(row)

def addDfRowANDOR(rows_list, concept_name, row_vals):
    """ Updates rows_list with values in row_vals. Note that there is a specific order to row_vals:
    [others, others_simp, rule_true, rule_total, concept_total, precision, recall, f1]"""
    row_keys = ['RULE for \''+concept_name+'\'', 'Simplified Rule', 'True Pos (tp)',
        'All Pos (tp+fp)', 'All Targets (tp+fn)', 'PRECISION', 'RECALL', 'F1 SCORE']
    addDfRow(rows_list, row_keys, row_vals)
    
def findConjunc(concept_name, concept_total, curr_lvl, rows_list, and_counts, target_combs, create_df):
    ''' Given a concept_name, a list of conjunctions (rules in curr_lvl), and a rows_list, 
    calculate its precision, recall, and F1 for predicting that concept_name. Add this information 
    to the rows_list.'''
    max_comb_size = max(target_combs.keys())
    for conjunc in curr_lvl: # for each conjunction in a list of conjunctions with the same size      
        # number of tp: images in which ruleAND is true and the concept is also in the image
        supersets_true = []
        for k in range(len(conjunc), max_comb_size+1):
            for comb in target_combs[k]:
                if set(conjunc).issubset(set(comb)):
                    supersets_true.append(comb)
        ruleAND_true = reduce(lambda x,y: x+y, [and_counts[p] for p in supersets_true]) # tp
        # number of pos: images in which ruleAND is true; the concept may or may not be also in the image
        ruleAND = set([w for w in conjunc if w != concept_name]) 
        supersets_total = []
        for k in range(len(ruleAND), max_comb_size+1):
            for comb in target_combs[k]:
                if set(ruleAND).issubset(set(comb)):
                    supersets_total.append(comb)
        ruleAND_total = reduce(lambda x,y: x+y, [and_counts[p] for p in supersets_total]) # tp+fp

        if ruleAND_true >= TRUE_CUTOFF:
            (precisionAND, recallAND) = calcPR(ruleAND_true, ruleAND_total, concept_total)

            othersAND = " & ".join(ruleAND)
            othersAND_simp = othersAND # no way to simplify ruleAND which is a set
            f1_AND = calcF1(precisionAND, recallAND) 

            if create_df:
                row_vals = [othersAND, othersAND_simp, ruleAND_true, 
                    ruleAND_total, concept_total, DISP.format(precisionAND), DISP.format(recallAND), 
                    DISP.format(f1_AND)]
                addDfRowANDOR(rows_list, concept_name, row_vals)

def mapLabelsToLetters(ruleOR):
    ''' This is a workaround that converts labels into letters for simplify_logic()
    (then convert back afterwards) because simplify_logic() has trouble parsing the word "sign"
    when it is a term.'''
    OR_letters_to_terms = {} # [letter] = term/label
    OR_terms_to_letters = {} # [term/label] = letter # opp of letters_to_terms
    ORletter_i = 0
    for term in set(ruleOR):
        letter = chr(ord('a')+ORletter_i)
        if term not in OR_letters_to_terms.values(): 
            # and therefore not in OR_terms_to_letters.keys() because the two dicts 
            # mirror each other
            OR_letters_to_terms[letter] = term
            OR_terms_to_letters[term] = letter
            ORletter_i+=1
    return (OR_letters_to_terms, OR_terms_to_letters)

def simplifyRule(othersOR_letters, simp_form):
    ''' Given an expression (othersOR_letters) and a simp_form (str), simplify the expression
    based on the specified simp_form. Return the simplified result and the number of terms in it.'''
    res = str(simplify_logic(othersOR_letters, simp_form))
    n = sum([len(w.split("&")) for w in \
        res.replace("(", '').replace(")", '').replace(" ", '').split("|")]) 
    return (res, n)

def findDisjunc(concept_name, concept_total, curr_lvl, rows_list, and_counts, create_df):
    ''' Given a concept_name, a list of conjunctions (rules in curr_lvl), and a rows_list, generate
    all possible disjunctions of the conjunctions in the list, and calculate the statistics on how 
    well each disjunction can be used to predict the presence of concept_name in an image. '''
    # Each group contains all possible disjunctions of the conjunctions in curr_lvl, arranged by the
    # size (number of conjunctions) in the disjunction as the keys. 
    # ex. "bed & pillow" | "bed & table" = ((bed, pillow), (bed, table)).
    # The terms/conjunctions all have the same size before the entire expr is simplified.
    groups = genPowerset(curr_lvl) 
    
    for n in range(2, max(groups.keys())+1): # group of 1 is already done in findConjunc()
        curr_group_lvl = groups[n]
        for group in curr_group_lvl: # group = a possible disjunction of conjunctions (part of powerset)
            # collect all the *nonunique* labels that aren't concept_name
            ruleOR = []
            for tup in group: # len(ruleOR) = num of labels in the rule
                ruleOR += [w for w in tup if w!=concept_name] 

            # assign a letter to represent each label (workaround for a bug by simpy.logic.simplify_logic())
            (OR_letters_to_terms, OR_terms_to_letters) = mapLabelsToLetters(ruleOR)
            
            # list of label tuples that were added to sumOR since it's inclusive
            ruleOR_true = 0
            ruleOR_total = 0
            for key in and_counts.keys(): # key = a conjunction
                # increment number of images for which ruleOR is true
                if any([set([w for w in tup if w!=concept_name]).issubset(key) for tup in group]): 
                    ruleOR_total += and_counts[key]
                # increment number of images for which ruleOR is true and the concept is present
                if any([set(tup).issubset(key) for tup in group]):
                    ruleOR_true += and_counts[key]

            if ruleOR_true >= TRUE_CUTOFF:
                (precisionOR, recallOR) = calcPR(ruleOR_true, ruleOR_total, concept_total)

                # glue the strings together: labels in a combination are joined by "&" and 
                # combinations are joined by "|"
                tupANDs = [("(" + (" & ".join([w for w in tup if w!=concept_name])) +")") for tup in group] 
                othersOR = " | ".join(tupANDs) # join the conjunctions, for both OR and XOR

                # replace the terms of the real rule with letters
                ORpattern = re.compile(r'\b(' + '|'.join(OR_terms_to_letters.keys()) + r')\b')
                othersOR_letters = str(ORpattern.sub(lambda x: OR_terms_to_letters[x.group()],\
                                                        othersOR))

                try: # find the shortest simplified version of the rule
                    (cnf_res, cnf_n) = simplifyRule(othersOR_letters, 'cnf')
                    (dnf_res, dnf_n) = simplifyRule(othersOR_letters, 'dnf')
                    othersOR_letters_simp = cnf_res if (cnf_n < dnf_n) else dnf_res

                    # replace the letters with the terms to form the simplified real rule
                    ORpattern_letters = re.compile(r'\b(' + '|'.join(OR_letters_to_terms.keys()) + r')\b')
                    othersOR_simp = str(ORpattern_letters.sub(lambda x: OR_letters_to_terms[x.group()], \
                        othersOR_letters_simp)) # the final simplified version of the rule

                    f1_OR = calcF1(precisionOR, recallOR) 

                    row_vals = [othersOR, othersOR_simp, ruleOR_true, ruleOR_total,
                                concept_total, DISP.format(precisionOR), DISP.format(recallOR), 
                                DISP.format(f1_OR)]
                    if create_df:
                        addDfRowANDOR(rows_list, concept_name, row_vals)
                except:
                    with open('doublecheck.txt', 'a') as f: # this requires postprocessing for unique lines
                        f.write(concept_name+'\n')

def modifyDfRow(d, concept_name, and_counts, target_combs):
    ''' This assumes a preexisting dictionary d that is partially filled. 
    It gets modified further in this dictionary.'''
    d['Concept'] = concept_name
    d['# Imgs'] = IMG_COUNTS[concept_name]
    d['Type'] = LABEL_TYPES[concept_name]
    d['Sort Method for Colabels'] = 'F1'
    d['Rule w/ Max F1'] = d['RULE for \''+concept_name+'\'']
    # the rest of the keys are either already in d, or will be filled in
    
    # We also want to check the hypothesis that, for a rule for predicting some concept,
    # if *none* of the labels in the rule is in some image, then that image actually 
    # does not contain the concept.
    # tn = this is true; 
    # fn = this is false, there is indeed a concept in the image despite the absence of those other labels;
    # unlabeled = some of the labels in the rule are in the image, so we're not sure whether concept is in the image.
    unique_concepts_in_rule = set() # this set will contain all labels in the rule (which would not contain the concept)
    for s in d['Simplified Rule'].replace("(", '').replace(")", '').replace(" ", '').split("|"):
        ss = s.split("&")
        unique_concepts_in_rule |= set(ss)
    not_implicit_keys = set() # keys that have any labels that are in the rule
    for uc in unique_concepts_in_rule:
        for lst in target_combs.values():
            for comb in lst:
                if uc in comb:
                    not_implicit_keys.add(comb)
    # groups of labels that *do not* have any labels in the rule
    all_combs = reduce(lambda x,y:x+y, target_combs.values())
    implicit_keys = set(all_combs) - not_implicit_keys

    # fam_n != IMG_COUNTS[c] because and_counts keep track of all related labels in the group, 
    # not just those that have c
    fam_n = sum(and_counts.values()) 
    # not even in the pool of AND_COUNTS because they have none of the related concepts/labels
    outsiders_n = OBJS_BY_IMG.size-fam_n 
    # number of imgs that do not contain any labels in ruleOR, nor concept_name
    tn = outsiders_n + sum([and_counts[k] for k in implicit_keys if concept_name not in k])
    d['Implicit True Negative (tn)'] = tn 
    # number of imgs that do not contain any labels in ruleOR, but contains concept_name
    fn = sum([and_counts[k] for k in implicit_keys if concept_name in k])
    d['Implicit False Negative (fn)'] = fn
    # number of imgs that contain some/all labels in ruleOR, and may or may not contain concept_name
    unlab = sum([and_counts[k] for k in not_implicit_keys]) 
    d['Implicit Unlabeled'] = unlab
    assert((tn+fn+unlab)==OBJS_BY_IMG.size)
    d['Implicit Accuracy (tn/(tn+fn))'] = "{0:.6f}".format(float(tn/(tn+fn)))

def getResultsANDOR(and_counts_dict, target_combs_dict, final_targets, create_df=True):
    """ Determines how well each rule (OR of ANDs) predicts each concept in [final_targets]. """
    
    if len(and_counts_dict) != len(final_targets):
        print("dict_counts_lst do not match final_targets -- please double check.")
        return []
        
    # this is returned even if create_df=False (in which case it's just empty)
    df_list = []
    total = []
    res_file_num = 0
    total_col_headers = ['Concept', '# Imgs', 'Type', 'Sort Method for Colabels', 
                        'Rule w/ Max F1', 'Simplified Rule', 'F1 SCORE', 'PRECISION','RECALL', 
                        'True Pos (tp)', 'All Pos (tp+fp)', 'All Targets (tp+fn)',
                        'Implicit True Negative (tn)', 'Implicit False Negative (fn)', 'Implicit Unlabeled',
                        'Implicit Accuracy (tn/(tn+fn))']
    
    concepts = [(t, ) for t in final_targets]
    for c in range(len(concepts)):
        concept = concepts[c] # tuple of (tiger, )
        and_counts = and_counts_dict[concept[0]] 
        target_combs = target_combs_dict[concept[0]]
            
        concept_name = concept[0] # "tiger"
        print(str(c+1)+"/"+str(len(concepts))+" of F1: "+concept[0])
        concept_total = sum([and_counts[key] for key in and_counts if concept_name in key]) # tp+fn

        # list of combs that have concept_name as well as other labels
        concept_multiples = [key for key in and_counts if (len(key)>1 and concept_name in key)] 
        
        # dict w/ val = list of label tuples (ones that don't just have the concept_name), 
        # key = length of each tuple in the list
        concept_tuples = {} 
        for m in concept_multiples:
            concept_tuples[len(m)] = concept_tuples.get(len(m), [])+[m]
            
        if create_df:
            rows_list = []
        # for each length of label tuple (tiger-orange = 2, tiger-orange-black = 3)
        for l in concept_tuples: 
            # concept_tuples = dict where keys = lengths
            curr_lvl = concept_tuples[l]
            # Treat a tuple (ex. ("bed", "pillow")) as a conjunction, modifies rows_list
            findConjunc(concept_name, concept_total, curr_lvl, rows_list, and_counts, target_combs, create_df)
            # Possibly with conjunctions as terms between |, modifies rows_list 
            findDisjunc(concept_name, concept_total, curr_lvl, rows_list, and_counts, create_df)

        if create_df:
            col_headers = ['RULE for \''+concept_name+'\'', 'Simplified Rule', \
                'PRECISION', 'RECALL', 'F1 SCORE', 'True Pos (tp)', 'All Pos (tp+fp)', \
                    'All Targets (tp+fn)']
            f1_rows_list = sorted(rows_list, key=lambda x: x['F1 SCORE'], reverse=True)
            df = pd.DataFrame(f1_rows_list, columns=col_headers)
            data_name = RESULTS_DIR+concept_name+'_results'
            df_name = data_name+'_F1.csv'
            df.to_csv(df_name)
            df_list.append(df)

            if f1_rows_list != []: # a rule exists for concept_name
                d = f1_rows_list[0]
                modifyDfRow(d, concept_name, and_counts, target_combs)
            else: # no rule for this concept_name that satisfies the requirements could be found
                d_vals = [concept_name, IMG_COUNTS[concept_name], LABEL_TYPES[concept_name], 'F1',\
                    '', '', -1, -2] + [-1]*8
                d = {k : v for (k, v) in zip(total_col_headers, d_vals)} 
            total.append(d)

            if len(total)==AUTOSAVE:
                totaldf = pd.DataFrame(total, columns=total_col_headers)
                res_file_num =int(((c+1)/AUTOSAVE))-1 # starts from 0
                total_data_name = RESULTS_DIR+'total_results_F1_'+str(res_file_num)+'.csv'
                totaldf.to_csv(total_data_name)
                total = [] # reset

    # for the ones after the last AUTOSAVE
    if total != []:
        res_file_num+=1
        totaldf = pd.DataFrame(total, columns=total_col_headers)
        total_data_name = RESULTS_DIR+'total_results_F1_'+str(res_file_num)+'.csv'
        totaldf.to_csv(total_data_name)
            
    return df_list

def genResults(concepts, special=[], start=0, end=5): 
    """ Given a list of concepts (strs), determine the rules that best predict each of them, and the 
    precision/recall/F1 scores of these rules. Sort the rules by decreasing order of F1 score."""
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # lst of lsts, w/ each inner list containing group of labels related to (and including) the corresponding concept
    final_concepts = []
    final_targets = [] 

    for i in range(len(concepts)):
        c = concepts[i]
        if IMG_COUNTS[c] >= CONCEPT_CUTOFF:
            # The following also writes the df (2nd output) of colabels in decreasing order of F1 score to csv
            (max_colabels, _) = getMaxCoLabels(c) 
            final_c = [] # group of labels of interest for given concept c
        
            # determine the colabels (a list for each concept) to study further
            if special==[]: # just get the first end-start number of colabels
                if len(max_colabels) >= (end-start): # We assume that max_colabels has 5 or more elements
                    final_c = [c]+[e[0] for e in max_colabels[start:end]] # e = tuple of (rule, freq, prec, rec)
                else:
                    final_c = []
            else: # look at special colabels
                final_c = [c]+special

            if len(max_colabels) >= 5:
                final_targets.append(final_c)
                final_concepts.append(c)

    #print('final_concepts:',final_concepts)
    #print('final_targets:',final_targets)
    
    # (list of dicts, dict, dict, list of dicts)
    (final_and_counts_dict, target_combs_dict) = countTargetLabels(OBJS_BY_IMG.size, final_targets)
    final_dfs = getResultsANDOR(final_and_counts_dict, target_combs_dict, final_concepts)

    print('len(final_dfs):', len(final_dfs))
    
    # return final_dfs

def loadJSON():
    ''' Load json files necessary for the data analysis. If none are available, generate them.
    We assume that the directory either have all 3 json files, or none of them. '''
    try:
        with open(DATASET_DIR+'ALL_LABEL_COUNTS.json', 'r') as ALL_LABEL_COUNTS_f:
            ALL_LABEL_COUNTS = json.load(ALL_LABEL_COUNTS_f)
        with open(DATASET_DIR+'IMG_COUNTS.json', 'r') as IMG_COUNTS_f:
            IMG_COUNTS = json.load(IMG_COUNTS_f)
        with open(DATASET_DIR+'LABEL_TYPES.json', 'r') as LABEL_TYPES_f:
            LABEL_TYPES = json.load(LABEL_TYPES_f)
    except FileNotFoundError:
        # The countAllCoLabels() function generates all 3 at the same time since it has to go 
        # through all the labels once to generate them.
        (ALL_LABEL_COUNTS, IMG_COUNTS, LABEL_TYPES) = countAllCoLabels(OBJS_BY_IMG, REL_DATA_LST)
        
        with open(DATASET_DIR+'ALL_LABEL_COUNTS.json', 'w') as ALL_LABEL_COUNTS_outfile:
            json.dump(ALL_LABEL_COUNTS, ALL_LABEL_COUNTS_outfile)
        with open(DATASET_DIR+'IMG_COUNTS.json', 'w') as IMG_COUNTS_outfile:
            json.dump(IMG_COUNTS, IMG_COUNTS_outfile)
        with open(DATASET_DIR+'LABEL_TYPES.json', 'w') as LABEL_TYPES_outfile:
            json.dump(LABEL_TYPES, LABEL_TYPES_outfile)
    return (ALL_LABEL_COUNTS, IMG_COUNTS, LABEL_TYPES)
    
# See countAllCoLabels():
# ALL_LABEL_COUNTS: keeps track of every label's tally of co-occurring labels;
# IMG_COUNTS: keeps track of every label's tally of images they've been assigned to;
# LABEL_TYPES: keeps track of every label's type (obj/attr/rel).
(ALL_LABEL_COUNTS, IMG_COUNTS, LABEL_TYPES) = loadJSON()
print('Finished loading data, including ALL_LABEL_COUNTS, IMG_COUNTS, and LABEL_TYPES.')

##############################################################
### Simulating automatedLabeling with results from Study 1 ###
##############################################################
def applyUserStudyData():    
    ''' Determine the 5 most commonly used words for each concept in the user study. 
    Output the best rules for predicting each concept in Visual Genome images based on boolean
    combinations of these words.'''
    study1_data = pd.read_csv('data/study1_data.csv')
    
    # study1_data.csv file is organized as follows: 
    # Each column starts with the concept in the first row. Subsequent rows consist of
    # words that participants mentioned when used to explain whether the concept was present 
    # in an image, and number of times they have done so. The rows are in decreasing order of number.
    # For example, first row is "Eating 1" (the number doesn't mean much; each concept
    # covers 5 columns); first two subsequent rows are "Food 111" and "person 64".
    study1_pair_counts = {}
    study1_pair_counts_raw = {}
    for index, row in study1_data.iterrows():
        for col in study1_data:
            k = col.split(" ")[0].lower()
            concept = ''
            if k == 'eating':
                concept = "eat"
            elif k == 'old':
                concept = 'old'
            elif k == 'nightstand':
                concept = 'nightstand'
            elif k == 'crossroad':
                concept = 'intersection'
            if type(row[col]) != float:
                [word, count] = row[col].split(" ")
                # print("synset conversion: (old, new)")
                    
                real_word = convertToSynset(word, OBJ_SYNSETS)
                #print((word, real_word)) # just making sure...
                if real_word.endswith('s'):
                    real_word = real_word[:-1]
                real_count = int(count)
                            
                if concept not in study1_pair_counts:
                    study1_pair_counts[concept] = {}
                if real_word not in study1_pair_counts[concept]:
                    study1_pair_counts[concept][real_word] = 0
                study1_pair_counts[concept][real_word] += real_count
                
                if concept not in study1_pair_counts_raw:
                    study1_pair_counts_raw[concept] = {}
                if word not in study1_pair_counts_raw[concept]:
                    study1_pair_counts_raw[concept][word] = 0

    final_concepts = []
    final_targets_orig = []
    final_targets = []
    for concept in study1_pair_counts:
        final_concepts.append(concept)
        sorted_counts = sorted(study1_pair_counts[concept], key=study1_pair_counts[concept].get, reverse=True)
        final_targets_orig.append([concept]+sorted_counts[:10])
    print(final_targets_orig) # purely for visual-examination purposes
    # This is the output of final_targets_orig:
    # [['eat', 'food', 'eating', 'person', 'mouth', 'plate', 'table', 'picture', 'eat', 'horse', 'fork'], 
    # ['old', 'old', 'look', 'phone', 'boat', 'photo', 'picture', 'paint', 'time', 'image', 'new'], 
    # ['nightstand', 'bed', 'nightstand', 'table', 'lamp', 'small', 'next', 'stand', 'bedroom', 'usually', 'room'], 
    # ['sidewalk', 'road', 'crossroad', 'street', 'sign', 'cro', 'stop', 'direction', 'traffic', 'track', 'image']]

    # We need to hardcode the final_targets because some of the most common colabels are repetitive (like
    # "eating" for the concept "eat") or nonsensical (ex. "cro" for the concept "crossing/intersection")
    for concept in final_concepts:
        # For each concept, take the 5 most commonly co-occurring words to form boolean expressions
        # to predict the presence of the concept in images
        if concept == 'eat':
            final_targets.append(['eat', 'food', 'person', 'mouth', 'plate', 'table'])
        elif concept == 'old':
            final_targets.append(['old', 'look', 'phone', 'boat', 'photo', 'picture'])
        elif concept == 'nightstand':
            final_targets.append(['nightstand', 'bed', 'table', 'lamp', 'small', 'next'])
        elif concept == 'intersection':
            final_targets.append(['intersection', 'road', 'street', 'sign', 'stop', 'direction'])
            
    os.makedirs(RESULTS_DIR, exist_ok=True)
    (final_and_counts_dict, target_combs_dict) = countTargetLabels(OBJS_BY_IMG.size, final_targets)
    final_dfs = getResultsANDOR(final_and_counts_dict, target_combs_dict, final_concepts) 

if args.concepts == 'all':
    all_labels = list(LABEL_TYPES.keys())
    genResults(all_labels)
elif args.concepts == 'userstudy-auto':
    study1 = ['nightstand','eat','intersection','old']
    genResults(study1)
elif args.concepts == 'test':
    genResults(['nightstand', 'bride'])
elif args.concepts == 'userstudy-manual':
    applyUserStudyData()