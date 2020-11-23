package io.kinference.algorithms.completion.suggest.feature

import io.kinference.algorithms.completion.suggest.CompletionInfo

class OneVariantFeatures(val context: String, val completion: CompletionInfo, val prefix: String = "") {

    companion object {
        const val prob = "prob"
        const val probs = "probs"
        const val mean_log_prob = "mean_log_prob"
        const val match_prefix = "match_prefix"
        const val prefix_matched_cnt = "prefix_matched_cnt"
        const val first_token_prob = "first_token_prob"
        const val max_prob = "max_prob"
        const val min_prob = "min_prob"
        const val tokens_len = "tokens_len"
        const val word_len = "word_len"
        const val symbol_len = "symbol_len"
        const val alphas_ration = "alphas_ration"
        const val context_len = "context_len"
        const val start_from_word = "start_from_word"
        const val is_repetition = "is_repetition"
    }

    private val features: MutableMap<String, Double> = HashMap()

    init {
        features[prob] = completion.info.probs.foldRight(1.0) { x, y -> x * y }
        features[mean_log_prob] = 1.0
    }
}
