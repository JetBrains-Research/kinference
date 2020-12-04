package io.kinference.algorithms.completion.suggest.collector

import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.CompletionModel
import io.kinference.algorithms.completion.generation.FairSeqGeneration
import io.kinference.algorithms.completion.generation.GenerationInfo
import io.kinference.algorithms.completion.generation.model.ModelWrapper
import io.kinference.algorithms.completion.tokenizer.BPETokenizer

/**
 * Completion generator that is using FairSeq beam-search under the hood.
 */
internal class FairSeqCompletionsGenerator(model: ModelWrapper, tokenizer: BPETokenizer) : BaseCompletionsGenerator(model, tokenizer) {
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

    private fun decodeSequences(sequences: List<List<GenerationInfo>>): List<List<CompletionModel.CompletionResult>> {
        val result = ArrayList<List<CompletionModel.CompletionResult>>(sequences.size)
        for (group in sequences) {
            result.add(group.map { CompletionModel.CompletionResult(tokenizer.decode(it.ids), it) })
        }
        return result
    }
}
