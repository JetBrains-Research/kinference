package io.kinference.algorithms.gec.tokenizer.subword.spacy

/**
 * TokenInfo token-level information about string
 * @param orth orthography string
 * @param lemma lemmatized string
 * @param tag string pos tag
 * @param norm normalized string
 */
data class SpacyTokenInfo(var orth: String, val lemma: String? = null, val tag: String? = null, val norm: String? = null)
