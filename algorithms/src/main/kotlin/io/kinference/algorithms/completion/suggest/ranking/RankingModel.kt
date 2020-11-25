package io.kinference.algorithms.completion.suggest.ranking

import io.kinference.algorithms.completion.CompletionModel

interface RankingModel {
    fun rank(context: String, prefix: String, completions: List<CompletionModel.CompletionResult>): List<CompletionModel.CompletionResult>
}


