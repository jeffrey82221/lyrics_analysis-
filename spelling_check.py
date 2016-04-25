# NOTE:
# Reference : misspell correct python with keyboard proximity
# photenic similarity
# possiblily using skip-gram


import enchant
from nltk.metrics import edit_distance

enchant.list_dicts()

britsh_spell_dict = enchant.Dict('en_GB')
american_spell_dict = enchant.Dict('en_US')




# NOTE:
# step one : check the correct vocs and the uncorrect vocs according to
# the enchant dictionary


def seperate_by_func(input_list, func):
    true_list = []
    false_list = []
    for element in input_list:
        if func(element) > 0:
            true_list.append(element)
        else:
            false_list.append(element)
    return true_list, false_list

voc_list=voc_dict[0].keys()
correct_vocs, uncorrect_vocs = seperate_by_func(
    voc_list, lambda x: britsh_spell_dict.check(x) or american_spell_dict.check(x))

# finding words that are britsh correct but american english not correct
# or vice. versa.

len(correct_vocs)
len(uncorrect_vocs)

# TODO:words that need to be futher checked and do recovering

hyphen_vocs, no_mark_vocs = seperate_by_func(
    uncorrect_vocs, lambda x: x.count('-'))
dash_vocs, no_mark_vocs = seperate_by_func(
    no_mark_vocs, lambda x: x.count('_'))
tilde_vocs, no_mark_vocs = seperate_by_func(
    no_mark_vocs, lambda x: x.count('~'))
period_vocs, no_mark_vocs = seperate_by_func(
    no_mark_vocs, lambda x: x.count('.'))
abbrev_quot_vocs, no_mark_vocs = seperate_by_func(
    no_mark_vocs, lambda x: x.count("'"))
backquot_mark_vocs, no_mark_vocs = seperate_by_func(
    no_mark_vocs, lambda x: x.count('`'))
backslash_vocs, no_mark_vocs = seperate_by_func(
    no_mark_vocs, lambda x: x.count('\\'))
slash_vocs, no_mark_vocs = seperate_by_func(
    no_mark_vocs, lambda x: x.count('/'))
star_mark_vocs, no_mark_vocs = seperate_by_func(
    no_mark_vocs, lambda x: x.count('*'))


len(hyphen_vocs)
len(dash_vocs)
len(tilde_vocs)
len(period_vocs)
len(abbrev_quot_vocs)
len(backquot_mark_vocs)
len(backslash_vocs)
len(slash_vocs)
len(star_mark_vocs)

# still 12303 vocs need to be distinguish
len(no_mark_vocs)

unambig_vocs = []
ambig_vocs = []
good_suggest_vocs = []
ng_suggest_vocs = []
case_error_vocs = []
case_error_suggest_vocs = []

long_voicing_vocs = []

theshold = 1
import string
for voc in no_mark_vocs:
    sug = list(set(britsh_spell_dict.suggest(
        voc) + american_spell_dict.suggest(voc)))

    if(len(sug) == 0):
        long_voicing_vocs.append(voc)
        # ng_suggest_vocs.append(None)
        continue
    else:
        sug_tuple = zip(map(lambda x: edit_distance(x, voc), sug), sug)
        sug_tuple = sorted(sug_tuple)
        # TODO:
        # here we need a way to sort the merged list, where the original order
        # of each list are remain, also ordered by the edition distance
        if(string.lower(sug_tuple[0][1]) == string.lower(voc)):
            case_error_vocs.append(voc)
            case_error_suggest_vocs.append(sug_tuple[0][1])
            continue
        else:
            if(len(sug) > 1):
                if(float(sug_tuple[1][0] - sug_tuple[0][0]) / sug_tuple[0][0] >= theshold):
                    unambig_vocs.append(voc)
                    good_suggest_vocs.append(sug_tuple[0][1])
                    continue
                else:
                    ambig_vocs.append(voc)
                    ng_suggest_vocs.append(sug_tuple[0][1])
                    continue
            else:  # len(sug)==1
                unambig_vocs.append(voc)
                good_suggest_vocs.append(sug_tuple[0][1])
                continue



# NOTE:Possible source of spelling error, already :
# 1.keybaord type error (introduce the keyboard proximity)
# 2.photenic error (using some kind of photenic similarity tool)
# 3.edition error (already in use, another kind of typing error)
# 4.phrase merging error (phrase)
# 5.case difference error (sometimes special names are not uppercase in
# the dictionary)


len(case_error_vocs)
zip(case_error_vocs, case_error_suggest_vocs)

unambig_voc_pairs = zip(unambig_vocs, good_suggest_vocs)
ambig_voc_pairs = zip(ambig_vocs, ng_suggest_vocs)
len(ambig_voc_pairs)
len(unambig_voc_pairs)


spliting_unambig_pairs, other_unambig_voc_pairs = seperate_by_func(
    unambig_voc_pairs, lambda x: string.lower(x[0]) == (''.join(x[1].split(' '))).lower())
spliting_ambig_pairs, other_ambig_voc_pairs = seperate_by_func(
    ambig_voc_pairs, lambda x: string.lower(x[0]) == (''.join(x[1].split(' '))).lower())

spliting_ambig_pairs
len(spliting_ambig_pairs)
len(spliting_unambig_pairs)


len(other_ambig_voc_pairs)
len(other_unambig_voc_pairs)

# NOTE:Edition distance still not suitable for check lyrics' spelling error 
# TODO:still need to find out more about :
other_unambig_voc_pairs
other_ambig_voc_pairs

# TODO:I can try photenic similarity to futher distinguish them
