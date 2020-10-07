package io.kinference.completion.suggest

import io.kinference.completion.generating.GenerationInfo

interface RankingModel {
    fun rank(context: String, prefix: String, completions: List<Pair<String, GenerationInfo>>): List<Pair<String, GenerationInfo>>
}

class FirstProbRankingModel() : RankingModel {
    override fun rank(context: String, prefix: String, completions: List<Pair<String, GenerationInfo>>): List<Pair<String, GenerationInfo>> {
        val comparatorBase = 100
        return completions.sortedBy { completion ->
//            val prob = Features.prob(completion.second)
            val firstProb = Features.prob(completion.second)
            val meanProb = Features.meanProb(completion.second)
            val prefixMatchedCount = Features.prefixMatchedCount(prefix, completion.first)
            firstProb * comparatorBase * comparatorBase + meanProb * comparatorBase + prefixMatchedCount
        }
    }

}
