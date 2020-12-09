package io.kinference.algorithms.completion.evaluation

import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.CompletionModel

class GolfMetric {
    fun compute(model: CompletionModel, text: String, context_len: Int = 50, config: CompletionConfig): Double {
        val score = GolfScore(text)

        for ((i, pair) in Metric.contextPrefixGenerator(text, "", context_len).withIndex()) {
            val (context, prefix) = pair
            val completions = model.complete(context, prefix, config)
            score.update(completions, prefix, i)
        }

        return score.get()
    }
}
