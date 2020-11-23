package io.kinference.algorithms.completion.suggest.ranking

import io.kinference.algorithms.completion.suggest.CompletionInfo

interface RankingModel {
    fun rank(context: String, prefix: String, completions: List<CompletionInfo>): List<CompletionInfo>
}


