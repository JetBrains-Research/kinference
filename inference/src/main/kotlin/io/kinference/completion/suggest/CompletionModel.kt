package io.kinference.completion.suggest

import io.kinference.completion.generating.GenerationInfo

class CompletionModel(private val completionsCollector: CompletionsCollector, private val rankingModel: RankingModel,
                      private val filterModel: FilterModel, private val postFilterModel: FilterModel? = null) {

    // @lru_cache(maxsize=200)
    private fun generate(context: String, prefix: String, min_len: Int, max_len: Int, num_beams: Int, repetition_penalty: Double, length_penalty: Double): List<Pair<String, GenerationInfo>> {
        return completionsCollector.collect(context, prefix, min_len, max_len, num_beams, repetition_penalty, length_penalty)
    }

    fun complete(context: String, prefix: String = "",
                 min_len: Int, max_len: Int, num_seqs: Int, num_beams: Int,
                 min_avg_log_prob: Double, min_prob: Double,
                 repetition_penalty: Double, length_penalty: Double): List<String> {
        if (context.trim().isEmpty()) {
            return listOf()
        }
        val trimmedContext = context.substring(context.length - 7000)

        var completions = generate(trimmedContext, prefix, min_len, max_len, num_beams, repetition_penalty, length_penalty)

        completions = filterModel.filter(trimmedContext, prefix, completions, min_avg_log_prob, min_prob)
        completions = rankingModel.rank(trimmedContext, prefix, completions)
        if (postFilterModel != null) {
            completions = postFilterModel.filter(trimmedContext, prefix, completions)
        }
        return completions.map { it.first }.subList(0, num_seqs)
    }
}
