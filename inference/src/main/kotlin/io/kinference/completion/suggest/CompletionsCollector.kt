package io.kinference.completion.suggest

import io.kinference.completion.BPETokenizer
import io.kinference.completion.generating.FairseqGeneration
import io.kinference.completion.generating.GenerationInfo
import io.kinference.completion.generating.ModelWrapper

interface CompletionsCollector {
    fun collect(context: String, prefix: String = "", minLen: Int, maxLen: Int, numBeams: Int,
                repetition_penalty: Double, lengthPenalty: Double): List<Pair<String, GenerationInfo>>

//    fun prediction_info(context: String, predictions: List<String>): List<GenerationInfo>
}

abstract class BaseCompletionsCollector(
    modelWrapperConstructor: (String, String) -> ModelWrapper,
    modelPath: String, baseModelName: String, val repetitionPenalty: Double = 1.0
) : CompletionsCollector {

    private val maxTokenizerLen = 1020
    protected val tokenizer = BPETokenizer(baseModelName, baseModelName) // TODO: fix path to tokenizer
    protected val languageModel = modelWrapperConstructor(modelPath, baseModelName)

    abstract fun generate(context: String, prefix: String = "", maxLen: Int = 1, numBeams: Int = 3,
                          repetition_penalty: Double? = null): List<Pair<String, GenerationInfo>>

    override fun collect(context: String, prefix: String, minLen: Int, maxLen: Int, numBeams: Int,
                repetition_penalty: Double, lengthPenalty: Double): List<Pair<String, GenerationInfo>> {
        if (context.trim().isEmpty()) {
            return emptyList()
        }

        val seenCompletions = HashSet<String>()
        val completions = generate(context, prefix, maxLen, numBeams, repetition_penalty)
        val result = ArrayList<Pair<String, GenerationInfo>>()

        for ((completion, gen_info) in completions) {

            // TODO: convert to one function?
            val trimmed = trimEnding(completion, gen_info)
            val (trimmedCompletion, trimmedGenInfo) = trimAfterSentenceEnd(trimmed.first, trimmed.second)

            val words = trimmedCompletion.split(' ')
            val targetLen = words.size

            if (targetLen < minLen) {
                continue
            }

            if (trimmedCompletion in seenCompletions) {
                continue
            }
            trimmedGenInfo.wordLen = targetLen
            seenCompletions.add(trimmedCompletion)

            result.add(Pair(trimmedCompletion, trimmedGenInfo))
        }

        return result
    }

    protected fun makeInputIds(context: String, maxLen: Int): List<Int> {
        var inputIds = tokenizer.encode(context)

        if (inputIds.size >= maxTokenizerLen - 2 - maxLen) {
            inputIds = inputIds.subList(inputIds.size - (maxTokenizerLen - 2 - maxLen), inputIds.size)
        }

        return inputIds
    }

    private fun trimEnding(completion: String, genInfo: GenerationInfo): Pair<String, GenerationInfo> {
        if (completion.isEmpty() || completion[completion.lastIndex].isLetterOrDigit()) {
            return Pair(completion, genInfo)
        }
        var i = 1
        while (i <= completion.length && !completion[completion.length - i].isLetterOrDigit()) {
            i += 1
        }
        i -= 1
        val codedAll = tokenizer.encode(completion)
        val codedTrimmed = tokenizer.encode(completion.substring(0, completion.length - i))

        var trimmedCompletion = completion
        if (codedTrimmed == codedAll.subList(0, codedTrimmed.size)) {
            trimmedCompletion = completion.substring(0, completion.length - i)
            genInfo.trim(codedTrimmed.size)
        }
        return Pair(trimmedCompletion, genInfo)
    }

    private fun trimAfterSentenceEnd(completion: String, genInfo: GenerationInfo): Pair<String, GenerationInfo> {
        if (completion.isEmpty()) {
            return Pair(completion, genInfo)
        }

        var i = 0
        while (i < completion.length && (completion[i].isLetterOrDigit() || completion[i] in " ,")) {
            i += 1
        }

        var trimmedCompletion = completion
        if (i < completion.length) {
            val codedAll = tokenizer.encode(completion)
            val codedTrimmed = tokenizer.encode(completion.substring(0, i))
            if (codedTrimmed == codedAll.subList(0, codedTrimmed.size)) {
                trimmedCompletion = completion.substring(0, i)
                genInfo.trim(codedTrimmed.size)
            }
        }

        return Pair(trimmedCompletion, genInfo)
    }

//    override fun prediction_info(context: String, predictions: List<String>): List<GenerationInfo> {
//        return listOf()
//    }
}

class FairseqCompletionsCollector(
    modelWrapperConstructor: (String, String) -> ModelWrapper,
    modelPath: String, baseModelName: String, private val numGroups: Int = 1, repetitionPenalty: Double = 1.0,
    prefixErrLimit: Int = 1, spellProb: Double = 0.02
) : BaseCompletionsCollector(modelWrapperConstructor, modelPath, baseModelName, repetitionPenalty) {

    private val beamSearch = FairseqGeneration(languageModel, tokenizer, prefixErrLimit, spellProb)

    override fun generate(context: String, prefix: String, maxLen: Int, numBeams: Int,
                          repetition_penalty: Double?): List<Pair<String, GenerationInfo>> {
//        repetition_penalty = repetition_penalty or self.repetition_penalty
//        num_groups = num_groups or self.num_groups

        val result = ArrayList<Pair<String, GenerationInfo>>()
        val inputIds = makeInputIds(context, maxLen)

        val completionsByLen = beamSearch.generate(
            inputIds, prefix, numBeams, maxLen, numGroups, repetition_penalty!!
        )
        for ((terminated, not_terminated) in completionsByLen) {
            val completions = decodeSequences(not_terminated)
            result.addAll(completions[0])
        }

        return result
    }

    private fun decodeSequences(sequences: List<List<Pair<List<Int>, GenerationInfo>>>): List<List<Pair<String, GenerationInfo>>> {
        val result: MutableList<List<Pair<String, GenerationInfo>>> = ArrayList()
        for (group in sequences) {
            val decodedStrings = group.map { tokenizer.decode(it.first) }
            result.add(group.mapIndexed { i, (_, info) -> Pair(decodedStrings[i], info) })
        }
        return result
    }
}
