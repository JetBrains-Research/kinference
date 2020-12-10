package io.kinference.algorithms.completion.evaluation

import io.kinference.algorithms.completion.CompletionModel
import kotlin.collections.ArrayList

abstract class Metric(val name: String) {
    companion object {
        fun contextPrefixGenerator(text: String, context: String = "", context_len: Int = 50): List<Pair<String, String>> {
            val result = ArrayList<Pair<String, String>>()
            val words = ArrayList<String>()
            var word = ""
//            val splitters = setOf(' ', '\n', '\t')
            val splitters = " \n\t"

//            if (context.isNotEmpty() && text.isNotEmpty() && splitters.contains(context[-1]) && splitters.contains(text[0])) {
//                context += '\n'
//            }

            result.add(Pair(context, ""))
            for (c in text) {
                if (splitters.contains(c)) {
                    if (words.size == context_len) {
                        words.removeAt(0)
                    }
                    if (word != "") {
                        words.add(word)
                    }
                    word = if (c == ' ') {
                        " "
                    } else {
                        words.add(c + "")
                        ""
                    }
                } else {
                    word += c
                }

                result.add(Pair(context + words.joinToString(""), word))
            }
            return result
        }
    }

    abstract fun compute(model: CompletionModel, data: List<Pair<String, String>>): Double
}
