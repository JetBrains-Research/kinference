package io.kinference.algorithms.gec.utils

import io.kinference.algorithms.gec.preprocessing.VerbsFormVocabulary

fun transformUsingVerb(token: String, form: String, verbsVocab: VerbsFormVocabulary): String{
    val formDict = verbsVocab.verbs2verbs[token]
    return if (formDict == null){
        token
    } else{
        val verb = formDict[form]
        verb ?: token
    }
}

fun transformUsingSplit(token: String): List<String>{
    return token.split("-")
}

fun transformUsingCase(token: String, case: String): String{
    if (case == "LOWER"){
        return token.toLowerCase()
    }
    else if (case == "UPPER"){
        return token.toUpperCase()
    }
    else if (case == "CAPITAL"){
        return token.capitalize()
    }
    else if (case == "CAPITAL_1"){
        return token
    }
    else if (case == "UPPER_-1"){
        return token
    }
    else{
        return token
    }
}

/*
def is_ascii(token: str) -> bool:
    return len(token) == len(token.encode())


def is_english(sentence: str):
    return langdetect.detect(sentence) == 'en'


def get_weights_name(transformer_name, lowercase):
    if transformer_name == 'bert' and lowercase:
        return 'bert-base-uncased'
    if transformer_name == 'bert' and not lowercase:
        return 'bert-base-cased'
    if transformer_name == 'distilbert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
            return 'distilbert-base-cased'
        else:
            return 'distilbert-base-uncased'
    if transformer_name == 'albert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'albert-base-v1'
    if lowercase:
        print('Warning! This model was trained only on cased sentences.')
    if transformer_name == 'roberta':
        return 'roberta-base'
    if transformer_name == 'gpt2':
        return 'gpt2'
    if transformer_name == 'transformerxl':
        return 'transfo-xl-wt103'
    if transformer_name == 'xlnet':
        return 'xlnet-base-cased'
    raise ValueError(f'Unknown transformer_name: {transformer_name}')


def transform_using_verb(token: str, form: str, verbs_vocab: VerbsFormVocabulary) -> str:
    forms_dict = verbs_vocab.verb2verbs.get(token)
    if forms_dict:
        verb = forms_dict.get(form)
        if verb:
            return verb
        else:
            return token
    else:
        return token


def transform_using_split(token: str) -> List[str]:
    return token.split("-")


def transform_using_plural(engine: inflect.engine, token: str, form: str) -> str:
    if form == "PLURAL":
        pl_noun = engine.plural_noun(token)
        return pl_noun if isinstance(pl_noun, str) else token
    elif form == "SINGULAR":
        sn_noun = engine.singular_noun(token)
        return sn_noun if isinstance(sn_noun, str) else token
    else:
        raise Exception(f"Unknown form {form}")


def transform_using_case(token: str, case: str) -> str:
    if case == "LOWER":
        return token.lower()
    elif case == "UPPER":
        return token.upper()
    elif case == "CAPITAL":
        return token.capitalize()
    elif case == "CAPITAL_1":
        return token[0] + token[1:].capitalize()
    elif case == "UPPER_-1":
        return token[:-1].upper() + token[-1]
    else:
        G_LOG.info(f'Unknown case: {case}')
        return token


def offset_calc(sent_ids: List[List[int]], offset_type: str) -> List[int]:
    word_lens = list(map(len, sent_ids))

    if offset_type == 'first':
        token_place_idxs = 1 + np.cumsum([0] + word_lens[:-1])
    # elif offset_type == 'first-last-mean':
    #     start = 1 + np.cumsum([0] + word_lens[:-1])
    #     end = np.cumsum([word_lens[0]] + word_lens[1:])
    #
    #     assert len(start) == len(end)
    #
    #     token_place_idxs = list(zip(start, end))
    else:
        raise NotImplementedError
    return token_place_idxs


def create_message_based_on_tag(tag: str):
    if tag == '$TRANSFORM_CASE_CAPITAL':
        return 'Capitalized first letter'
    elif tag.startswith('$TRANSFORM_VERB'):
        return 'Incorrect verb form'
    elif tag in ['$TRANSFORM_AGREEMENT_PLURAL', '$TRANSFORM_AGREEMENT_SINGULAR']:
        return 'Incorrect word form'
    elif tag == '$DELETE_SPACES':
        return 'Rudimentary spaces'
    elif tag.startswith('$APPEND_') or tag.startswith('$REPLACE_'):
        last = tag.split('_')[-1]
        if last in punctuation:
            return 'Incorrect punctuation'
        elif last in ['a', 'an', 'the']:
            return 'Article error'
        else:
            return 'Grammatical error'
    else:
        return 'Grammatical error'


def create_message_based_on_error_type(error_type):
    if not error_type.endswith('OTHER'):

        if error_type.endswith('ADJ'):
            if error_type.startswith('M'):
                return 'Missed adjective'
            elif error_type.startswith('R'):
                return 'Adjective replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary adjective'
            else:
                return 'Incorrect adjective'

        elif error_type.endswith('ADJ:FORM'):
            if error_type.startswith('R'):
                return 'Adjective form replacement'
            else:
                return 'Incorrect adjective form'

        elif error_type.endswith('ADV'):
            if error_type.startswith('M'):
                return 'Missed adverb'
            elif error_type.startswith('R'):
                return 'Adverb replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary adverb'
            else:
                return 'Incorrect adverb'

        elif error_type.endswith('CONJ'):
            if error_type.startswith('M'):
                return 'Missed conjunction'
            elif error_type.startswith('R'):
                return 'Conjunction replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary conjunction'
            else:
                return 'Incorrect conjunction'

        elif error_type.endswith('CONTR'):
            if error_type.startswith('M'):
                return 'Missed contraction'
            elif error_type.startswith('R'):
                return 'Contraction replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary contraction'
            else:
                return 'Incorrect contraction'

        elif error_type.endswith('DET'):
            if error_type.startswith('M'):
                return 'Missed determiner'
            elif error_type.startswith('R'):
                return 'Determiner replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary determiner'
            else:
                return 'Incorrect determiner'

        elif error_type.endswith('MORPH'):
            if error_type.startswith('R'):
                return 'Morphology replacement'
            else:
                return 'Incorrect morphology'

        elif error_type.endswith('NOUN'):
            if error_type.startswith('M'):
                return 'Missed noun'
            elif error_type.startswith('R'):
                return 'Noun replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary noun'
            else:
                return 'Incorrect noun'

        elif error_type.endswith('NOUN:INFL'):
            if error_type.startswith('R'):
                return 'Noun inflection replacement'
            else:
                return 'Incorrect noun inflection'

        elif error_type.endswith('NOUN:NUM'):
            if error_type.startswith('R'):
                return 'Noun form replacement'
            else:
                return 'Incorrect noun form'

        elif error_type.endswith('NOUN:POSS'):
            if error_type.startswith('M'):
                return 'Missed possessive noun form'
            elif error_type.startswith('R'):
                return 'Possessive noun form replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary possessive noun form'
            else:
                return 'Incorrect possessive noun form'

        elif error_type.endswith('ORTH'):
            if error_type.startswith('R'):
                return 'Orthography replacement'
            else:
                return 'Incorrect orthography'

        elif error_type.endswith('PART'):
            if error_type.startswith('M'):
                return 'Missed particle'
            elif error_type.startswith('R'):
                return 'Particle replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary particle'
            else:
                return 'Incorrect particle'

        elif error_type.endswith('PREP'):
            if error_type.startswith('M'):
                return 'Missed preposition'
            elif error_type.startswith('R'):
                return 'Preposition replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary preposition'
            else:
                return 'Incorrect preposition'

        elif error_type.endswith('PRON'):
            if error_type.startswith('M'):
                return 'Missing pronoun'
            elif error_type.startswith('R'):
                return 'Pronoun replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary pronoun'
            else:
                return 'Incorrect pronoun'

        elif error_type.endswith('PUNCT'):
            if error_type.startswith('M'):
                return 'Missing punctuation'
            elif error_type.startswith('R'):
                return 'Punctuation replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary punctuation'
            else:
                return 'Incorrect punctuation'

        elif error_type.endswith('SPELL'):
            return 'Spelling error'

        elif error_type.endswith('VERB'):
            if error_type.startswith('M'):
                return 'Missed verb'
            elif error_type.startswith('R'):
                return 'Verb replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary verb'
            else:
                return 'Incorrect verb'

        elif error_type.endswith('VERB:FORM'):
            if error_type.startswith('M'):
                return 'Missed verb form'
            elif error_type.startswith('R'):
                return 'Verb form replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary verb form'
            else:
                return 'Incorrect verb form'

        elif error_type.endswith('VERB:INFL'):
            if error_type.startswith('R'):
                return 'Verb inflection replacement'
            else:
                return 'Verb inflection error'
        elif error_type.endswith('VERB:SVA'):
            if error_type.startswith('R'):
                return 'Subject-verb form replacement'
            else:
                return 'Subject-verb agreement error'

        elif error_type.endswith('VERB:TENSE'):
            if error_type.startswith('M'):
                return 'Missed verb tense form'
            elif error_type.startswith('R'):
                return 'Verb tense form replacement'
            elif error_type.startswith('U'):
                return 'Unnecessary verb tense form'
            else:
                return 'Misapplication of tense morphology'

        elif error_type.endswith('WO'):
            return 'Word order error'

        elif error_type.endswith('@SPACE_DELETE'):
            return 'Rudimentary spaces'
    else:
        return None


def calculate_tokens_borders_and_with_spaces(text: str, tokens: List[str], text_with_space: bool = False) -> Tuple[List[Tuple[int, int]], List[bool]]:
    """ Calculates character based indices of token in `text` and whether each token has a preceding space in `text`.
    First token would have preceding space either if it has a preceding space in `text` or `text_with_space` is True. """
    token_start_end: List[Tuple[int, int]] = []
    with_spaces: List[bool] = []
    start_from = 0
    for idx, token in enumerate(tokens):
        start_idx = text.find(token, start_from)
        assert start_idx != -1, f'Can find token "{token}" in "{text}" | start_from: {start_from}'

        token_start_end.append((start_idx, start_idx + len(token)))
        if idx == 0 and text_with_space:
            with_spaces.append(True)
        else:
            with_spaces.append(start_idx >= start_from + 1)

        start_from = start_idx + len(token)
    return token_start_end, with_spaces
 */
