package io.kinference.algorithms.completion.suggest

import io.kinference.algorithms.completion.config.Config
import io.kinference.algorithms.completion.config.GenerationConfig
import io.kinference.algorithms.completion.generation.*
import io.kinference.algorithms.completion.generation.model.GPT2ModelWrapper
import io.kinference.algorithms.completion.generation.search.Search
import io.kinference.algorithms.completion.tokenizer.BPETokenizer

interface CompletionsCollector {
    fun collect(context: String, prefix: String, config: GenerationConfig): List<CompletionInfo>

//    fun prediction_info(context: String, predictions: List<String>): List<GenerationInfo>
}

abstract class BaseCompletionsCollector(config: Config) : CompletionsCollector {
    private val maxTokenizerLen = config.tokenizer.maxSeqLen - 4
    protected val languageModel = GPT2ModelWrapper(config.loader, config.model)
    protected val tokenizer = BPETokenizer(config.loader)

    abstract fun generate(context: String, prefix: String, config: GenerationConfig): List<CompletionInfo>

    override fun collect(context: String, prefix: String, config: GenerationConfig): List<CompletionInfo> {
        if (context.trim().isEmpty()) {
            return emptyList()
        }

        val seenCompletions = HashSet<String>()
        val completions = generate(context, prefix, config)
        val result = ArrayList<CompletionInfo>()

        for (completion in completions) {

            // TODO: convert to one function?
            val trimmedCompletion = completion.trimEnding().trimAfterSentenceEnd()

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

    private fun CompletionInfo.trimEnding(): CompletionInfo {
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
        return CompletionInfo(trimmedCompletion, this.info)
    }

    private fun CompletionInfo.trimAfterSentenceEnd(): CompletionInfo {
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

        return CompletionInfo(trimmedCompletion, this.info)
    }
}

class FairseqCompletionsCollector(config: Config) : BaseCompletionsCollector(config) {

    private val beamSearch = FairSeqGeneration(languageModel, tokenizer)

    override fun generate(context: String, prefix: String, config: GenerationConfig): List<CompletionInfo> {
        val result = ArrayList<CompletionInfo>()
        val inputIds = makeInputIds(context, config.maxLen)

        val completionsByLen = beamSearch.generate(inputIds, prefix, config)
        for (completionsGroup in completionsByLen) {
            val completions = decodeSequences(completionsGroup)
            result.addAll(completions[0])
        }

        return result
    }

    private fun decodeSequences(sequences: List<List<Search.HypothesisInfo>>): List<List<CompletionInfo>> {
        val result: MutableList<List<CompletionInfo>> = ArrayList()
        for (group in sequences) {
            val decodedStrings = group.map { tokenizer.decode(it.hypothesis) }
            result.add(group.mapIndexed { i, (_, info) -> CompletionInfo(decodedStrings[i], info) })
        }
        return result
    }
}
