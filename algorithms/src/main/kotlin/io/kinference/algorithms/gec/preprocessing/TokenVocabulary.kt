package io.kinference.algorithms.gec.preprocessing

/**
 * Vocabulary for tokens/tags
 */
@Suppress("unused")
data class TokenVocabulary(val token2index: MutableMap<String, Int> = HashMap(), val index2token: MutableMap<Int, String> = HashMap()) {
    companion object {
        fun load(file: String): TokenVocabulary {
            val vocab = TokenVocabulary()
            for (line in file.lines().filter { it.isNotBlank() }) {
                vocab.addToken(line)
            }
            return vocab
        }
    }

    fun addToken(token: String) {
        if (token2index.containsKey(token)) return

        val index = token2index.keys.size
        token2index[token] = index
        index2token[index] = token
    }

    fun addTokens(tokens: List<String>) {
        for (token in tokens) {
            addToken(token)
        }
    }

    fun getTokenIndex(token: String): Int {
        return token2index[token]!!
    }

    fun getTokenByIndex(idx: Int): String = index2token[idx]!!

    fun getIndexToTokenVocab(): Map<Int, String> = index2token

    fun getTokenToIndexVocab(): Map<String, Int> = token2index

    fun size(): Int = token2index.size
}

/**
 * Vocbalulary for forms of verbs
 */
data class VerbsFormVocabulary(val verbs2verbs: MutableMap<String, MutableMap<String, String>> = HashMap()) {
    companion object {
        fun load(file: String): VerbsFormVocabulary {
            val vocab = VerbsFormVocabulary()

            for (line in file.lines().filter { it.isNotBlank() }) {
                val (verb, form) = line.split(':', limit = 2)
                val (inVerb, outVerb) = verb.split('_', limit = 2)
                vocab.addVerb(inVerb, form, outVerb)
            }
            return vocab
        }
    }

    fun addVerb(inVerb: String, form: String, outVerb: String) {
        if (verbs2verbs.containsKey(inVerb)) {
            verbs2verbs[inVerb]?.set(form, outVerb)
        } else {
            verbs2verbs[inVerb] = mutableMapOf(form to outVerb)
        }
    }
}
