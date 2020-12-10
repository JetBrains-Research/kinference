package io.kinference.algorithms.completion.evaluation

import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.CompletionModel

/**
 * Golf metric. Min actions to write text with completion / text length
 */
class GolfMetric {
    fun compute(model: CompletionModel, text: String, contextLen: Int = 50, config: CompletionConfig): Double {
        val score = GolfScore(text)

        for ((i, pair) in Metric.contextPrefixGenerator(text, "", contextLen).withIndex()) {
            val (context, prefix) = pair
            val completions = model.complete(context, prefix, config)
            score.update(completions, prefix, i)
        }

        return score.get()
    }
}
