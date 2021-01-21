package io.kinference.algorithms.gec.classifier

import io.kinference.algorithms.gec.tokenizer.utils.CharUtils

object GECClassifier {
    fun classifyError(tag: String): String {
        when {
            tag == "\$TRANSFORM_CASE_CAPITAL" -> {
                return "Capitalized first letter"
            }
            tag.startsWith("\$TRANSFORM_VERB") -> {
                return "Incorrect verb form"
            }
            tag in listOf("\$TRANSFORM_AGREEMENT_PLURAL", "\$TRANSFORM_AGREEMENT_SINGULAR") -> {
                return "Incorrect word form"
            }
            tag == "\$DELETE_SPACES" -> {
                return "Rudimentary spaces"
            }
            tag.startsWith("\$APPEND_") or tag.startsWith("\$REPLACE_") -> {
                val last = tag.split('_').last()
                return when {
                    CharUtils.isPunctuation(last.toCharArray().last()) -> {
                        "Incorrect punctuation"
                    }
                    last in listOf("a", "an", "the") -> {
                        "Article error"
                    }
                    else -> {
                        "Grammatical error"
                    }
                }
            }
            else -> {
                return "Grammatical error"
            }
        }
    }

}
