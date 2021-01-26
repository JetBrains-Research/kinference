package io.kinference.algorithms.gec.changes

import io.kinference.algorithms.gec.preprocessing.VerbsFormVocabulary
/** Some default transformations that are used by grammar error correction module */
object Transformations {
    fun transformUsingVerb(token: String, form: String, verbsVocab: VerbsFormVocabulary): String {
        return verbsVocab.verbs2verbs[token]?.get(form) ?: token
    }

    fun transformUsingSplit(token: String): List<String> {
        return token.split("-")
    }

    fun transformUsingCase(token: String, case: String): String {
        when (case) {
            "LOWER" -> return token.toLowerCase()
            "UPPER" -> return token.toUpperCase()
            "CAPITAL" -> return token.capitalize()
            "CAPITAL_1" -> {
                val first = token.take(1)
                val rst = token.drop(1).capitalize()
                return first + rst
            }
            "UPPER_-1" -> {
                val last = token.takeLast(1)
                val rst = token.dropLast(1).toUpperCase()
                return rst + last
            }
            else -> return token
        }
    }
}
