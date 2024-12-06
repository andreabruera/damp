import re

def transform_basic_word(word):
    versions = [word.lower(), word.capitalize()]
    return versions

def transform_italian_word(word):
    word = word.lower()
    versions = [word]
    accents = ['à', 'è', 'ò', 'ù', 'ì']
    '''
    if word[-1] in accents:
        versions.append(word[:-1])
        if 'à' in word:
            versions.append(word.replace('à', 'a'))
        if 'è' in word:
            versions.append(word.replace('è', 'e'))
        if 'ì' in word:
            versions.append(word.replace('ì', 'i'))
        if 'ò' in word:
            versions.append(word.replace('ò', 'o'))
        if 'ù' in word:
            versions.append(word.replace('ù', 'u'))
    '''
    final_versions = list()
    for w in versions:
        final_versions.append(w)
        #final_versions.append(w.capitalize())
    return final_versions

def transform_german_word(word, lowercase=True):
    word = word.lower()
    word = re.sub('^ein\s|^eine\s|^der\s|^das\s|^die\s|^ne\s|^dann\s', '', word)
    word = re.sub('^e\s', 'e-', word)
    versions = [word]
    original_versions = [word]
    substitutions = [
                     ('ae', 'ä'),
                     ('oe', 'ö'),
                     ('ue', 'ü'),
                     ('ss', 'ß'),
                     ]
    for word in original_versions:
        ### collecting different versions of a word
        for forw, back in substitutions:
            if forw in word:
                new_versions = [w for w in versions]
                for w in new_versions:
                    corr_word = w.replace(forw, back)
                    versions.append(corr_word)
            if back in word:
                new_versions = [w for w in versions]
                for w in new_versions:
                    corr_word = w.replace(back, forw)
                    versions.append(corr_word)
    if not lowercase:
        versions = set(
                       ### capitalized
                       [' '.join([tok.capitalize() for tok in w.split()]) for w in versions] +\
                       [' '.join([tok.capitalize() for tok in word.split()])] + \
                       []
                       ### non-capitalized
                       #[w for w in versions]
                       )
    else:
        versions = set(
                       ### capitalized
                       [' '.join([tok.capitalize() for tok in w.split()]) for w in versions] +\
                       [' '.join([tok.capitalize() for tok in word.split()])] + \
                       ### non-capitalized
                       [w for w in versions]
                       )

    #print(versions)
    return versions
