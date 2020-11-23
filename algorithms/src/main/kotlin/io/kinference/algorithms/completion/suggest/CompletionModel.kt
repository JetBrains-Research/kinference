package io.kinference.algorithms.completion.suggest

import io.kinference.algorithms.completion.config.Config
import io.kinference.algorithms.completion.config.GenerationConfig
import io.kinference.algorithms.completion.generation.GenerationInfo
import io.kinference.algorithms.completion.suggest.filtering.FilterModel
import io.kinference.algorithms.completion.suggest.ranking.RankingModel
import kotlin.math.max
import kotlin.math.min

data class CompletionInfo(val text: String, val info: GenerationInfo)

class CompletionModel(private val completionsCollector: CompletionsCollector, private val rankingModel: RankingModel,
                      private val filterModel: FilterModel, private val postFilterModel: FilterModel? = null) {

    // @lru_cache(maxsize=200)
    private fun generate(context: String, prefix: String, config: GenerationConfig): List<CompletionInfo> {
        return completionsCollector.collect(context, prefix, config)
    }

    fun complete(context: String, prefix: String = "", config: Config): List<String> {
        val trimmedContext = context.substring(max(0, context.length - 7000))

        var completions = generate(trimmedContext, prefix, config.generation)

        completions = filterModel.filter(trimmedContext, prefix, completions, config.filter)
        completions = rankingModel.rank(trimmedContext, prefix, completions)
        if (postFilterModel != null) {
            completions = postFilterModel.filter(trimmedContext, prefix, completions, config.filter)
        }
        return completions.take(min(config.numSeqs, completions.size)).map { it.text }
    }
}
