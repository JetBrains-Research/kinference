package io.kinference.algorithms.completion.suggest.collector

import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.CompletionModel
import io.kinference.algorithms.completion.generation.model.ModelWrapper
import io.kinference.algorithms.completion.tokenizer.BPETokenizer

/**
 * Base class for all completions generators that trims and cleans up completions
 * got from beam-search or other implementation before passing it to the client.
 */
internal abstract class BaseCompletionsGenerator(internal val model: ModelWrapper,
                                                 internal val tokenizer: BPETokenizer) : CompletionsGenerator {
    private val maxTokenizerLen = model.maxSeqLen - 4

    protected abstract fun generateWithSearch(context: String, prefix: String, config: CompletionConfig.Generation): List<CompletionModel.CompletionResult>

    override fun generate(context: String, prefix: String, config: CompletionConfig.Generation): List<CompletionModel.CompletionResult> {
        if (context.isBlank()) return emptyList()

        val seenCompletions = HashSet<String>()
        val completions = generateWithSearch(context, prefix, config)
        val result = ArrayList<CompletionModel.CompletionResult>()

        for (completion in completions) {
            // TODO: convert to one function?
            val trimmedCompletion = completion.trimEnding().trimAfterSentenceEnd()
            if (trimmedCompletion.text.isEmpty()
                || trimmedCompletion.text.length == 1 && !completion.text[0].isLetterOrDigit()
                || !tokenizer.isValidString(trimmedCompletion.text)
            ) continue

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
