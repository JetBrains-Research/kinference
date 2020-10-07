package io.kinference.completion.suggest

import io.kinference.completion.FilterConfig
import io.kinference.completion.generating.GenerationInfo
import kotlin.math.pow

interface FilterModel {
    fun filter(context: String, prefix: String, completions: List<Pair<String, GenerationInfo>>,
               config: FilterConfig): List<Pair<String, GenerationInfo>>
}

class ProbFilterModel : FilterModel {
    override fun filter(context: String, prefix: String, completions: List<Pair<String, GenerationInfo>>,
                        config: FilterConfig): List<Pair<String, GenerationInfo>> {
        return completions.filter { completion ->
            val prob = Features.prob(completion.second)
            val meanProb = Features.meanProb(completion.second)
            val startFromWord = completion.first[0] == ' ' && completion.first[1].isLetter()
            val symbolLen = completion.first.length - prefix.length
//            val is_repetition = isRepetition(completion.first, context)

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
