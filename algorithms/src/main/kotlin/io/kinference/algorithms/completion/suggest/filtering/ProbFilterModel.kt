package io.kinference.algorithms.completion.suggest.filtering

import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.CompletionModel
import io.kinference.algorithms.completion.suggest.feature.Features
import kotlin.math.pow

class ProbFilterModel : FilterModel {
    override fun filter(context: String, prefix: String, completions: List<CompletionModel.CompletionResult>, config: CompletionConfig.Filter): List<CompletionModel.CompletionResult> {
        return completions.filter { completion ->
            val prob = Features.prob(completion.info)
            val meanProb = Features.meanProb(completion.info)
            val startFromWord = completion.text[0] == ' ' && completion.text[1].isLetter()
            val symbolLen = completion.text.length - prefix.length
//            val is_repetition = isRepetition(io.kinference.algorithms.completion.first, context)

            meanProb >= config.minAvgLogProb
                && prob >= config.minProb
                && symbolLen >= config.minSymbolLen
                && startFromWord
//                && is_repetition
        }
    }

    private fun isRepetition(completion: String, context: String): Boolean {
        val compWords = completion.trim().split(' ')
        val contextWords = context.trim().split(' ')
        var compCnt = 0
        var contextCnt = 0.0
        for (i in compWords.indices) {
            val word = compWords[i]
            for (j in compWords.indices) {
                val otherWord = compWords[j]
                if (i != j && word == otherWord) {
                    compCnt += 1
                }
            }
            for (j in contextWords.indices) {
                val otherWord = contextWords[j]
                if (word == otherWord) {
                    contextCnt += 1.3.pow(-(contextWords.size - j - 1 + i).toDouble())
                }
            }
        }
        return compCnt + contextCnt > 1
    }
}
