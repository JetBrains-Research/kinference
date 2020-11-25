package io.kinference.algorithms.completion.suggest.collector

import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.CompletionModel
import io.kinference.algorithms.completion.generation.FairSeqGeneration
import io.kinference.algorithms.completion.generation.search.Search

/**
 * Completion generator that is using FairSeq beam-search under the hood.
 */
class FairSeqCompletionsGenerator(config: CompletionConfig) : BaseCompletionsGenerator(config) {
    private val beamSearch = FairSeqGeneration(model, tokenizer)

    override fun generateWithSearch(context: String, prefix: String, config: CompletionConfig.Generation): List<CompletionModel.CompletionResult> {
        val result = ArrayList<CompletionModel.CompletionResult>()
        val inputIds = makeInputIds(context, config.maxLen)

        val completionsByLen = beamSearch.generate(inputIds, prefix, config)
        for (completionsGroup in completionsByLen) {
            val completions = decodeSequences(completionsGroup)
            result.addAll(completions[0])
        }

        return result
    }

    private fun decodeSequences(sequences: List<List<Search.HypothesisInfo>>): List<List<CompletionModel.CompletionResult>> {
        val result = ArrayList<List<CompletionModel.CompletionResult>>(sequences.size)
        for (group in sequences) {
            val decodedStrings = group.map { tokenizer.decode(it.hypothesis) }
            result.add(group.mapIndexed { i, (_, info) -> CompletionModel.CompletionResult(decodedStrings[i], info) })
        }
        return result
    }
}
