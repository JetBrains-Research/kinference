package io.kinference.algorithms.completion.evaluation

import java.lang.Double.min

internal class GolfScore(val text: String) {
    private val scores = DoubleArray(text.length + 20) { Double.MAX_VALUE }
    private var prevScores = DoubleArray(text.length + 20)

    init {
        scores[0] = 0.0
        scores[1] = 1.0
    }

    fun update(completions: List<String>, prefix: String, i: Int) {
        prevScores = scores.clone()

        scores[i + 2] = min(scores[i + 2], scores[i + 1] + 1)
        for ((pos, completion) in completions.withIndex()) {
            var completionSuffixLen = completion.length - prefix.length
            if (completion == prefix + text.substring(i, Integer.min(text.length, i + completionSuffixLen))) {
                scores[i + 1 + completionSuffixLen] = min(scores[i + 1 + completionSuffixLen], scores[i + 1] + pos + 1)
            }
            val words = completion.split(' ')
            for ((word_ind, _) in words.withIndex()) {
                val completionPrefix = words.subList(0, word_ind + 1).joinToString(" ")
                if (completionPrefix.isEmpty()) {
                    continue
                }
                completionSuffixLen = completionPrefix.length - prefix.length
                if (completionPrefix == text.substring(i, Integer.min(text.length, i + completionSuffixLen))) {
                    scores[i + 1 + completionSuffixLen] = min(scores[i + 1 + completionSuffixLen], scores[i + 1] + pos + word_ind + 1)
                } else {
                    break
                }
            }
        }
    }

    fun improvement(): Double {
        var res = 0.0
        for ((prev_score, score) in prevScores.zip(scores)) {
            if (prev_score != score) {
                res += 1
            }
        }
        return res
    }

    fun get(): Double {
        return scores[text.length] / text.length
    }

    fun getActionsNum(): Double {
        return scores[text.length]
    }
}
