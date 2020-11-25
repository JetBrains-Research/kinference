package io.kinference.algorithms.completion.suggest.collector

import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.CompletionModel

/**
 * Generator of completions for specific context and prefix.
 * Under the hood implementation may use beam search or other techniques to get suggestions from GPT-like model
 */
interface CompletionsGenerator {
    /**
     * Perform generation of completions from specific [context] and [prefix] with [config] configuration
     *
     * Note, that completions would start from [prefix]
     *
     * Also, it is expected that context would not have trailing whitespaces and prefix would have leading whitespaces
     * or even consist of only whitespaces.
     */
    fun generate(context: String, prefix: String, config: CompletionConfig.Generation): List<CompletionModel.CompletionResult>
}


