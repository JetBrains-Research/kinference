package io.kinference.algorithms.completion

import io.kinference.algorithms.completion.generation.GenerationInfo
import io.kinference.algorithms.completion.suggest.collector.CompletionsGenerator
import io.kinference.algorithms.completion.suggest.filtering.FilterModel
import io.kinference.algorithms.completion.suggest.ranking.RankingModel
import kotlin.math.min


/**
 * Model that performs natural language completion based on context and prefix
 *
 * Under the hood, model would use GPT-like model and beam-search depending
 * on [generator] use
 *
 * @param generator used to get completions
 * @param ranking used to rank completions got from generator
 * @param preFilter would be used to filter completions before ranking
 */
class CompletionModel(
    private val generator: CompletionsGenerator,
    private val ranking: RankingModel,
    private val preFilter: FilterModel
) {
    /** Result of completion generation -- text and metadata (probabilities, etc.) */
    data class CompletionResult(val text: String, val info: GenerationInfo)

    private fun generate(context: String, prefix: String, config: CompletionConfig.Generation): List<CompletionResult> {
        return generator.generate(context, prefix, config)
    }

    /**
     * Get completions for [context] and [prefix] with respect to [config] configuration
     *
     * Note, that complete expects that context would not have trailing whitespaces and prefix
     * would have leading whitespaces (or even consist of only whitespaces).
     *
     * So, if you have something like `Hello wo` you should split it into `context: "Hello"`
     * and `prefix: " wo"`
     */
    fun complete(context: String, prefix: String, config: CompletionConfig): List<String> {
        var completions = generate(context, prefix, config.generation)

        completions = preFilter.filter(context, prefix, completions, config.filter)
        completions = ranking.rank(context, prefix, completions)

        return completions.take(min(config.numSuggestions, completions.size)).map { it.text }
    }
}
