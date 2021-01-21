package io.kinference.algorithms.gec.corrector

import io.kinference.algorithms.gec.GECTag
import io.kinference.algorithms.gec.corrector.correction.SentenceCorrections
import io.kinference.algorithms.gec.corrector.correction.TextCorrection
import io.kinference.algorithms.gec.encoder.PreTrainedTextEncoder
import io.kinference.algorithms.gec.postprocessing.GecPostprocessor
import io.kinference.algorithms.gec.preprocessing.*
import io.kinference.algorithms.gec.utils.offsetCalc
import kotlin.math.min

/**
 * Main class for correction generation.
 * @param model  Seq2Logits class which can predict tags for each token in sequence
 * @param encoder is text encoder used by model of GEC
 * @param labelsVocab label vocabulary
 * @param dTagsVocab detection tags vocabulary
 * @param verbsVocab verbs form vocabulary
 * @param preprocessor sequence preprocessor
 * @param postprocessor correction postprocessor
 * @param iterations quantity of iterations which model should do for returning corrections
 * @param minCorrectionProb minimum probability for correction
 * @param minErrorProb minimum probability for error
 * @param confidence bias for not-changing strategy
 *
 */
class GECCorrector(
    private val model: Seq2Logits,
    private val encoder: PreTrainedTextEncoder,
    labelsVocab: TokenVocabulary,
    dTagsVocab: TokenVocabulary,
    private val verbsVocab: VerbsFormVocabulary,
    private val preprocessor: GecPreprocessor,
    private val postprocessor: GecPostprocessor,
    val iterations: Int = 3,
    minCorrectionProb: Double = 0.0,
    minErrorProb: Double = 0.0,
    confidence: Double = 0.0
) {
    private val tagger = GECTagger(
        model = model,
        encoder = encoder,
        labelsVocabulary = labelsVocab,
        dTagsVocabulary = dTagsVocab,
        minCorrectionProb = minCorrectionProb,
        minErrorProb = minErrorProb,
        confidence = confidence
    )

    /**
     * data class for Correction result from each iteration
     */
    internal data class CorrectionResult(val sentence: String, val corrections: SentenceCorrections)

    /**
     * calculate correction for batch sentence for evaluation
     */
    internal fun toCorrectedSentences(sentences: List<String>): List<CorrectionResult> {
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
        // val correctedSentence = postprocessor.postprocess(sentCorrections)
        return textCorrections
    }

    /**
     * generation feature vectors for sentence
     */
    private fun generateTaggerFeatures(sentObj: SentenceCorrections, tokens: List<SentenceCorrections.GECToken>): List<GECTaggerFeatures> {
        val tokSent = tokens.map { it.text }

        val encodedTokens = tokens.filter { it.isUsed }.map { it.encoded }

        val features = ArrayList<GECTaggerFeatures>()
        val modelMaxLen = 512   // TODO(Add to tokenizer field ModelMaxLength)
        for (sliceStart in encodedTokens.indices step modelMaxLen) {
            val sliceEnd = min(a = sliceStart + modelMaxLen, b = encodedTokens.size)
            val encodedTokensSlice = encodedTokens.subList(fromIndex = sliceStart, toIndex = sliceEnd)
            val offsets = offsetCalc(encodedTokensSlice, "first")

            val flatTokens = listOf(encoder.clsId) + encodedTokensSlice.flatten() + listOf(encoder.sepId)

            features.add(
                GECTaggerFeatures(
                    sent = sentObj.sent,
                    sentId = sentObj.sentId,
                    tokSent = tokSent,
                    encTokSent = encodedTokens,
                    flatEncSent = flatTokens,
                    offset = offsets
                )
            )
        }
        return features
    }

    /**
     * calculate SentenceCorrections for each correction iteration
     */
    private fun calculateSentenceCorrections(sentence: String): SentenceCorrections {
        val sentCorrections = preprocessor.preprocess(sentId = 0, sentence = sentence)
        for (idx in 0 until iterations) {
            val tokens: List<SentenceCorrections.GECToken> = sentCorrections.toCorrectedTokenSentence().filter { token -> token.isUsed }
            val taggerFeatures = generateTaggerFeatures(sentCorrections, tokens)
            val tagged = taggerFeatures.map { tagger.correct(it) }

            if (tagged.size > 1) return sentCorrections

            sentCorrections.addTokenToCorrections(
                tokenSentence = tokens,
                taggedSentence = tagged[0],
                encoder = encoder,
                verbsFormVocabulary = verbsVocab
            )
            updateFinalIteration(sentObj = sentCorrections, taggedSentence = tagged[0])
            if (sentCorrections.isCorrect) break
        }
        return sentCorrections
    }

    /**
     * calculate list SentenceCorrection for each correction iteration
     */
    private fun calculateSentenceCorrectionsList(sentences: List<String>): List<SentenceCorrections> {
        val result = sentences.mapIndexed { index, sentence -> preprocessor.preprocess(index, sentence) }
        for (idx in 0 until iterations) {
            val iterationCorrections = result.filter { c -> !c.isCorrect }
            val tokensDict = iterationCorrections.associate { it.sentId to it.toCorrectedTokenSentence().filter { token -> token.isUsed }.toList() }
            val preprocessedSentences = ArrayList<GECTaggerFeatures>()

            for (corrections in iterationCorrections) {
                val features = generateTaggerFeatures(corrections, tokens = tokensDict[corrections.sentId]!!)
                if (features.size == 1) {
                    preprocessedSentences.add(features[0])
                }
            }

            val taggedSentences = tagger.correctList(preprocessedSentences, batchSize = 20)

            for (tagged in taggedSentences) {
                result[tagged.sentId].addTokenToCorrections(
                    tokenSentence = tokensDict[tagged.sentId]!!,
                    taggedSentence = tagged,
                    encoder = encoder,
                    verbsFormVocabulary = verbsVocab
                )
                updateFinalIteration(result[tagged.sentId], tagged)
            }

            if (result.all { it.isCorrect }) {
                break
            }
        }
        return result
    }

    /** Check whether iterations should be stopped for this sentence */
    private fun updateFinalIteration(sentObj: SentenceCorrections, taggedSentence: GECTagger.TagSentObject) {
        if (taggedSentence.tags.any { it != GECTag.KEEP.value && it != GECTag.UNK.value }) {
            return
        }

        sentObj.isCorrect = true
    }
}
