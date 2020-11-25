package io.kinference.algorithms.completion.suggest.filtering

import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.CompletionModel

interface FilterModel {
    fun filter(context: String, prefix: String, completions: List<CompletionModel.CompletionResult>, config: CompletionConfig.Filter): List<CompletionModel.CompletionResult>
}
