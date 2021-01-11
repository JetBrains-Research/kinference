package io.kinference.algorithms.gec.corrector

import io.kinference.algorithms.gec.postprocessing.GecPostprocessor
import io.kinference.algorithms.gec.preprocessing.*
import io.kinference.algorithms.gec.utils.*
import kotlin.math.min

/**
 * data class for Correction result from each iteration
 */
data class CorrectionResult(val sentence: String, val corrections: SentenceCorrections)

/**
 * Main class for correction generation.
 * @param model
 * @param textProcessor
 * @param labelsVocab
 * @param dTagsVocab
 * @param verbsVocab
 * @param preprocessor
 * @param postprocessor
 * @param iterations
 * @param minCorrectionProb
 * @param minErrorProb
 * @param confidence
 *
 */
class GECCorrector(val model: Seq2Logits,
                   private val textProcessor: TransformersTextprocessor,
                   labelsVocab: Vocabulary,
                   dTagsVocab: Vocabulary,
                   private val verbsVocab: VerbsFormVocabulary,
                   private val preprocessor: GecPreprocessor,
                   private val postprocessor: GecPostprocessor,
                   val iterations: Int = 3,
                   minCorrectionProb: Double = 0.0,
                   minErrorProb: Double = 0.0,
                   confidence: Double = 0.0) {
    private val tagger = GecTagger(
        model = model,
        textProcessor = textProcessor,
        labelsVocabulary = labelsVocab,
        dTagsVocabulary = dTagsVocab,
        minCorrectionProb = minCorrectionProb,
        minErrorProb = minErrorProb,
        confidence = confidence
    )

    /**
     * calculate correction for batch sentence for evaluation
     */
    fun toCorrectedSentences(sentences: List<String>): List<CorrectionResult> {

        val corrections = calculateSentenceCorrectionsList(sentences)
        return corrections.map { CorrectionResult(sentence = postprocessor.postprocess(it), corrections = it) }
    }

    /**
     * calculate correction for one sentence
     */
    fun evaluateCorrectionsInFewIterations(sentence: String): List<TextCorrection> {
        val sentCorrection = calculateSentenceCorrections(sentence)
        return toTextCorrections(sentCorrection)
    }

    /**
     * calculate corrections for batch of sentence
     */
    fun evaluateCorrectionsListInFewIterations(sentences: List<String>): List<List<TextCorrection>> {
        val sentCorrections = calculateSentenceCorrectionsList(sentences)
        return sentCorrections.map { toTextCorrections(it) }
    }

    /**
     * textCorrection generation
     */
    private fun toTextCorrections(sentCorrections: SentenceCorrections): List<TextCorrection> {

        val textCorrections = sentCorrections.toTextCorrections()
        val correctedSentence = postprocessor.postprocess(sentCorrections)
        return textCorrections
    }

    /**
     * generation feature vectors for sentence
     */
    private fun generateTaggerFeatures(sentObj: SentenceCorrections, tokens: List<Token>): List<GecTaggerFeatures> {
        val tokSent = tokens.map { it.text }
        val encodedTokens = ArrayList<List<Int>>()

        for (token in tokens) {
            if (token.isUsed) {
                assert(token.encodedData != null)
                encodedTokens.add(token.encodedData)
            }
        }

        val features = ArrayList<GecTaggerFeatures>()
        val modelMaxLen = 512   // TODO(Add to tokenizer field ModelMaxLength)
        for (sliceStart in 0 until encodedTokens.size step modelMaxLen) {
            val sliceEnd = min(a = sliceStart + modelMaxLen, b = encodedTokens.size)
            val encodedTokensSlice = encodedTokens.subList(fromIndex = sliceStart, toIndex = sliceEnd)
            val offsets = offsetCalc(encodedTokensSlice, "first")
            var flatTokens: List<Int> = encodedTokensSlice.flatten()

            flatTokens = listOf(textProcessor.tokenizer.clsId) + flatTokens + listOf(textProcessor.tokenizer.sepId)

            features.add(GecTaggerFeatures(
                sent = sentObj.sent,
                sentId = sentObj.sentId,
                tokSent = tokSent,
                encTokSent = encodedTokens,
                flatEncSent = flatTokens,
                offset = offsets))
        }
        return features
    }

    /**
     * calculate SentenceCorrections for each correction iteration
     */
    private fun calculateSentenceCorrections(sentence: String): SentenceCorrections {
        val sentCorrections = preprocessor.preprocess(sentId = 0, sentence = sentence)
        for (idx in 0 until iterations) {
            val tokens: List<Token> = sentCorrections.toCorrectedTokenSentence().filter { token -> token.isUsed }
            val taggerFeatures = generateTaggerFeatures(sentCorrections, tokens)
            val tagged = taggerFeatures.map { tagger.correct(it) }
            if (tagged.size > 1) {
                return sentCorrections
            }

            sentCorrections.addTokenToCorrections(
                tokenSentence = tokens,
                taggedSentence = tagged[0],
                textProcessor = textProcessor,
                verbsFormVocabulary = verbsVocab
            )
            if (isFinalIteration(sentObj = sentCorrections, taggedSentence = tagged[0])) {
                break
            }
        }
        return sentCorrections
    }

    /**
     * calculate list SentenceCorrection for each correction iteration
     */
    private fun calculateSentenceCorrectionsList(sentences: List<String>): List<SentenceCorrections> {
        val correctionList = sentences.mapIndexed { index, sentence -> preprocessor.preprocess(index, sentence) }
        for (idx in 0 until iterations) {
            val iterationCorrections = correctionList.filter { c -> !c.isCorrect }
            val tokensDict = iterationCorrections.map { it.sentId to it.toCorrectedTokenSentence().filter { token -> token.isUsed }.toList() }.toMap()
            val preprocessedSentences = ArrayList<GecTaggerFeatures>()

            for (corrections in iterationCorrections) {
                val features = generateTaggerFeatures(corrections, tokens = tokensDict[corrections.sentId]!!)
                if (features.size == 1) {
                    preprocessedSentences.add(features[0])
                }
            }

            val taggedSentences = tagger.correctList(preprocessedSentences, batchSize = 20)

            for (tagged in taggedSentences) {
                correctionList[tagged.sentId].addTokenToCorrections(
                    tokenSentence = tokensDict[tagged.sentId]!!,
                    taggedSentence = tagged,
                    textProcessor = textProcessor,
                    verbsFormVocabulary = verbsVocab)
                isFinalIteration(correctionList[tagged.sentId], tagged)
            }

            if (correctionList.all { it.isCorrect }) {
                break
            }
        }
        return correctionList
    }

    /**
     * bool for check is final iteration
     */
    private fun isFinalIteration(sentObj: SentenceCorrections, taggedSentence: TagSentObject): Boolean {
        for (tag in taggedSentence.tags) {
            if (tag != TagKeep.value) {
                return false
            }
        }
        sentObj.isCorrect = true
        return true
    }
}
