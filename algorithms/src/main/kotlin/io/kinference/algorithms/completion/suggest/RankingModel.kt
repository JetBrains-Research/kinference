package io.kinference.algorithms.completion.suggest

interface RankingModel {
    fun rank(context: String, prefix: String, completions: List<CompletionInfo>): List<CompletionInfo>
}

class FirstProbRankingModel : RankingModel {
    override fun rank(context: String, prefix: String, completions: List<CompletionInfo>): List<CompletionInfo> {
        return completions.sortedBy { completion ->
            val firstProb = Features.prob(completion.info)
            val meanProb = Features.meanProb(completion.info)
            val prefixMatchedCount = Features.prefixMatchedCount(prefix, completion.text)
            firstProb * COMPARATOR_SQUARE + meanProb * COMPARATOR_BASE + prefixMatchedCount
        }
    }

    companion object {
        private const val COMPARATOR_BASE = 100
        private const val COMPARATOR_SQUARE = COMPARATOR_BASE * COMPARATOR_BASE
    }

}
