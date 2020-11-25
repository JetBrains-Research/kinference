package io.kinference.algorithms.completion

import io.kinference.algorithms.completion.generation.GenerationInfo
import io.kinference.algorithms.completion.suggest.CompletionsCollector
import io.kinference.algorithms.completion.suggest.filtering.FilterModel
import io.kinference.algorithms.completion.suggest.ranking.RankingModel
import kotlin.math.min


class CompletionModel(
    private val collector: CompletionsCollector,
    private val ranking: RankingModel,
    private val preFilter: FilterModel,
    private val postFilter: FilterModel? = null
) {
    data class CompletionResult(val text: String, val info: GenerationInfo)

    private fun generate(context: String, prefix: String, config: CompletionConfig.Generation): List<CompletionResult> {
        return collector.collect(context, prefix, config)
    }

    fun complete(context: String, prefix: String, config: CompletionConfig): List<String> {
        var completions = generate(context, prefix, config.generation)

        completions = preFilter.filter(context, prefix, completions, config.filter)
        completions = ranking.rank(context, prefix, completions)
        completions = postFilter?.filter(context, prefix, completions, config.filter) ?: completions

        return completions.take(min(config.numSuggestions, completions.size)).map { it.text }
    }
}
