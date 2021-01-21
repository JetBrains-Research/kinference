package io.kinference.algorithms.completion

import io.kinference.algorithms.completion.generation.model.GPT2ModelWrapper
import io.kinference.algorithms.completion.suggest.collector.FairSeqCompletionsGenerator
import io.kinference.algorithms.completion.suggest.filtering.ProbFilterModel
import io.kinference.algorithms.completion.suggest.ranking.WordTrieIterativeGolfRanking
import io.kinference.algorithms.completion.tokenizer.BPETokenizer

/**
 * Factory for creation Completion Models
 */
object CompletionModelFactory {
    /**
     * Factory for creation Completion Models
     *
     * @param config for model and tokenizer loading
     */
    fun createCompletionModel(config: CompletionConfig): CompletionModel {
        val tokenizer = BPETokenizer(config.loader)
        val model = GPT2ModelWrapper(config.loader, config.model)

        val generator = FairSeqCompletionsGenerator(model, tokenizer)
        val ranking = WordTrieIterativeGolfRanking(tokenizer, config.numSuggestions, -1000.0)
        val preFilter = ProbFilterModel()

        return CompletionModel(generator, ranking, preFilter)
    }
}
