package io.kinference.algorithms.completion.suggest.filtering

import io.kinference.algorithms.completion.config.FilterConfig
import io.kinference.algorithms.completion.suggest.CompletionInfo

interface FilterModel {
    fun filter(context: String, prefix: String, completions: List<CompletionInfo>, config: FilterConfig): List<CompletionInfo>
}
