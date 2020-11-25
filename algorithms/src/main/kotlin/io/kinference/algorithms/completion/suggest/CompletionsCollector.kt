package io.kinference.algorithms.completion.suggest

import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.CompletionModel
import io.kinference.algorithms.completion.generation.FairSeqGeneration
import io.kinference.algorithms.completion.generation.model.GPT2ModelWrapper
import io.kinference.algorithms.completion.generation.search.Search
import io.kinference.algorithms.completion.tokenizer.BPETokenizer

interface CompletionsCollector {
    fun collect(context: String, prefix: String, config: CompletionConfig.Generation): List<CompletionModel.CompletionResult>
}

abstract class BaseCompletionsCollector(config: CompletionConfig) : CompletionsCollector {
    private val maxTokenizerLen = config.tokenizer.maxSeqLen - 4
    protected val languageModel = GPT2ModelWrapper(config.loader, config.model)
    protected val tokenizer = BPETokenizer(config.loader)

    abstract fun generate(context: String, prefix: String, config: CompletionConfig.Generation): List<CompletionModel.CompletionResult>

    override fun collect(context: String, prefix: String, config: CompletionConfig.Generation): List<CompletionModel.CompletionResult> {
        if (context.trim().isEmpty()) {
            return emptyList()
        }

        val seenCompletions = HashSet<String>()
        val completions = generate(context, prefix, config)
        val result = ArrayList<CompletionModel.CompletionResult>()

        for (completion in completions) {

            // TODO: convert to one function?
            val trimmedCompletion = completion.trimEnding().trimAfterSentenceEnd()
            if (trimmedCompletion.text.isEmpty() || trimmedCompletion.text.length == 1 && !completion.text[0].isLetterOrDigit()) continue

            val words = trimmedCompletion.text.trim().split(' ')
            val targetLen = words.size

            if (targetLen < config.minLen || trimmedCompletion.text in seenCompletions) continue
            trimmedCompletion.info.wordLen = targetLen

            seenCompletions.add(trimmedCompletion.text)
            result.add(trimmedCompletion)
        }

        return result
    }

    protected fun makeInputIds(context: String, maxLen: Int): IntArray {
        var inputIds = tokenizer.encode(context)

        if (inputIds.size >= maxTokenizerLen - 2 - maxLen) {
            inputIds = inputIds.copyOfRange(inputIds.size - (maxTokenizerLen - 2 - maxLen), inputIds.size)
        }

        return inputIds
    }

    private fun CompletionModel.CompletionResult.trimEnding(): CompletionModel.CompletionResult {
        if (this.text.isEmpty() || this.text[text.lastIndex].isLetterOrDigit()) {
            return this
        }
        var i = 1
        while (i <= this.text.length && !this.text[text.length - i].isLetterOrDigit()) i++
        i--
        val codedAll = tokenizer.encode(this.text)
        val codedTrimmed = tokenizer.encode(this.text.substring(0, this.text.length - i))

        var trimmedCompletion = this.text
        if (codedTrimmed.contentEquals(codedAll.copyOfRange(0, codedTrimmed.size))) {
            trimmedCompletion = trimmedCompletion.substring(0, trimmedCompletion.length - i)
            this.info.trim(codedTrimmed.size)
        }
        return CompletionModel.CompletionResult(trimmedCompletion, this.info)
    }

    private fun CompletionModel.CompletionResult.trimAfterSentenceEnd(): CompletionModel.CompletionResult {
        if (this.text.isEmpty()) {
            return this
        }

        var i = 0
        while (i < this.text.length && (this.text[i].isLetterOrDigit() || this.text[i] in " ,")) i++

        var trimmedCompletion = this.text
        if (i < this.text.length) {
            val codedAll = tokenizer.encode(this.text)
            val codedTrimmed = tokenizer.encode(this.text.substring(0, i))
            if (codedTrimmed.contentEquals(codedAll.copyOfRange(0, codedTrimmed.size))) {
                trimmedCompletion = this.text.substring(0, i)
                this.info.trim(codedTrimmed.size)
            }
        }

        return CompletionModel.CompletionResult(trimmedCompletion, this.info)
    }
}

class FairseqCompletionsCollector(config: CompletionConfig) : BaseCompletionsCollector(config) {

    private val beamSearch = FairSeqGeneration(languageModel, tokenizer)

    override fun generate(context: String, prefix: String, config: CompletionConfig.Generation): List<CompletionModel.CompletionResult> {
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
        val result: MutableList<List<CompletionModel.CompletionResult>> = ArrayList()
        for (group in sequences) {
            val decodedStrings = group.map { tokenizer.decode(it.hypothesis) }
            result.add(group.mapIndexed { i, (_, info) -> CompletionModel.CompletionResult(decodedStrings[i], info) })
        }
        return result
    }
}
