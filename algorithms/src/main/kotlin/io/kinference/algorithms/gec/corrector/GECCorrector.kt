package io.kinference.algorithms.gec.corrector

import io.kinference.algorithms.gec.postprocessing.GecCorrectionPostprocessor
import io.kinference.algorithms.gec.postprocessing.GecPostprocessor
import io.kinference.algorithms.gec.preprocessing.*
import io.kinference.algorithms.gec.utils.*
import kotlin.math.min

data class CorrectionResult(val sentence: String, val corrections: SentenceCorrections)

class GECCorrector(val model: Seq2Logits,
                   val textProcessor: TransformersTextprocessor,
                   val labelsVocab: Vocabulary,
                   val dTagsVocab: Vocabulary,
                   val verbsVocab: VerbsFormVocabulary,
                   val preprocessor: GecPreprocessor,
                   val postprocessor: GecPostprocessor,
                   val iterations: Int = 3,
                   val useSpellcheckerFirst: Boolean = false,
                   val minCorrectionProb: Double = 0.0,
                   val minErrorProb: Double = 0.0,
                   val confidence: Double = 0.0)
{
    val tagger = GecTagger(
        model = model,
        textProcessor = textProcessor,
        labelsVocabulary = labelsVocab,
        dTagsVocabulary = dTagsVocab,
        minCorrectionProb = minCorrectionProb,
        minErrorProb = minErrorProb,
        confidence = confidence
    )

    fun toCorrectedSentences(sentences: List<String>): List<CorrectionResult>{

        val corrections = calculateSentenceCorrectionsList(sentences)
        return corrections.map { CorrectionResult(sentence = postprocessor.postprocess(it), corrections = it) }
    }

    fun evaluateCorrectionsInFewIterations(sentence: String): List<TextCorrection>{
        val sentCorrection = calculateSentenceCorrections(sentence)
        return toTextCorrections(sentCorrection)
    }

    fun evaluateCorrectionsListInFewIterations(sentences: List<String>): List<List<TextCorrection>>{
        val sentCorrections = calculateSentenceCorrectionsList(sentences)
        return sentCorrections.map { toTextCorrections(it) }
    }

    private fun toTextCorrections(sentCorrections: SentenceCorrections): List<TextCorrection>{

        val textCorrections = sentCorrections.toTextCorrections()
        val correctedSentence = postprocessor.postprocess(sentCorrections)
        return textCorrections
    }

    fun generateTaggerFeatures(sentObj: SentenceCorrections, tokens: List<Token>): List<GecTaggerFeatures>{
        val tokSent = tokens.map { it.text }
        val encodedTokens = ArrayList<List<Int>>()

        for (token in tokens){
            if (token.isUsed){
                assert(token.encodedData != null)
                encodedTokens.add(token.encodedData)
            }
        }

        val features = ArrayList<GecTaggerFeatures>()
        val modelMaxLen = 512   // TODO(Add to tokenizer field ModelMaxLength)
        for (sliceStart in 0 until encodedTokens.size step modelMaxLen){
            val sliceEnd = min(a = sliceStart + modelMaxLen, b = encodedTokens.size)
            val encodedTokensSlice= encodedTokens.subList(fromIndex = sliceStart, toIndex = sliceEnd)
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

    private fun calculateSentenceCorrections(sentence: String): SentenceCorrections{
        val sentCorrections = preprocessor.preprocess(sentId = 0, sentence = sentence)
        for (idx in 0 until iterations){
            val tokens: List<Token> = sentCorrections.toCorrectedTokenSentence().filter { token -> token.isUsed }
            val taggerFeatures = generateTaggerFeatures(sentCorrections, tokens)
            val tagged = taggerFeatures.map { tagger.correct(it) }
            if (tagged.size > 1){
                return sentCorrections
            }

            sentCorrections.addTokenToCorrections(
                tokenSentence = tokens,
                taggedSentence = tagged[0],
                textProcessor = textProcessor,
                verbsFormVocabulary = verbsVocab
            )
            if (isFinalIteration(sentObj = sentCorrections, taggedSentence = tagged[0])){
                break
            }
        }
        return sentCorrections
    }

    private fun calculateSentenceCorrectionsList(sentences: List<String>): List<SentenceCorrections>{
        val correctionList = sentences.mapIndexed { index, sentence -> preprocessor.preprocess(index, sentence) }
        for (idx in 0 until iterations){
            val iterationCorrections = correctionList.filter { c -> !c.isCorrect }
            val tokensDict = iterationCorrections.map { it.sentId to it.toCorrectedTokenSentence().filter { token -> token.isUsed }.toList() }.toMap()
            val preprocessedSentences = ArrayList<GecTaggerFeatures>()

            for (corrections in iterationCorrections){
                val features = generateTaggerFeatures(corrections, tokens = tokensDict[corrections.sentId]!!)
                if (features.size == 1){
                    preprocessedSentences.add(features[0])
                }
            }

            val taggedSentences = tagger.correctList(preprocessedSentences, batchSize = 20)

            for (tagged in taggedSentences){
                correctionList[tagged.sentId].addTokenToCorrections(
                    tokenSentence = tokensDict[tagged.sentId]!!,
                    taggedSentence = tagged,
                    textProcessor = textProcessor,
                    verbsFormVocabulary = verbsVocab)
                isFinalIteration(correctionList[tagged.sentId], tagged)
            }

            if (correctionList.all { it.isCorrect }){
                break
            }
        }
        return correctionList
    }

    private fun isFinalIteration(sentObj: SentenceCorrections, taggedSentence: TagSentObject): Boolean{
        for (tag in taggedSentence.tags){
            if (tag != tagKeep.value){
                return false
            }
        }
        sentObj.isCorrect = true
        return true
    }
}

fun main(){
    val model = Seq2Logits("/Users/Ivan.Dolgov/ivandolgov/model&test/model.onnx")
    val textProcessor: TransformersTextprocessor = TransformersTextprocessor("bert-base-uncased")
    val labelsVocab: Vocabulary = Vocabulary.loadFromFile("/Users/Ivan.Dolgov/ivandolgov/projects/grazie-ml/grazie/gec/config/labels.txt")
    val dTagsVocab: Vocabulary = Vocabulary.loadFromFile("/Users/Ivan.Dolgov/ivandolgov/projects/grazie-ml/grazie/gec/config/d_tags.txt")
    val verbsVocab: VerbsFormVocabulary = VerbsFormVocabulary.setupVerbsFormVocab("/Users/Ivan.Dolgov/ivandolgov/projects/grazie-ml/grazie/gec/config/verb-form-vocab.txt")
    val preprocessor: GecPreprocessor = GecCorrectionPreprocessor(textProcessor = textProcessor, useStartToken = true, useForEval = false)
    val postprocessor: GecPostprocessor = GecCorrectionPostprocessor()

    val gecCorrector = GECCorrector(model, textProcessor, labelsVocab, dTagsVocab, verbsVocab, preprocessor, postprocessor, iterations = 5)
    val testStrings = listOf(
        "It are bad .",
        "It are even more interesting .",
        "What I sure about is the fact that I am going to need an official certificate in order to prove that I 've studied the language .",
        "I am easy going person with a lot of empathy for children .",
        "I study English because I love it and I 'd like to speak very fluently and watch a movie without a subtitles .",
        "I 'll hopes it does not involve inconvenience to you .",
        "And I likes cake too .",
        "It are bad things to do and I think it is a good thing s to do it and",
        "I 've change the place and I do n't knows which one is the best for me .",
        "As I 'm new here , I 'm lost and do n't know where is to my hotel .",
        "You do n't need to worry about your writing skills any more , improving you 're text has never be more easier !",
        "It are an friend .",
        "I have travelled around most of European countries and I be able to understanding , talking and writin highly three languages : Spanish , English and French .",
        "I 'll hopes it does not involve inconvenience to you .",
        "It is give to the body the energy as well as regular work of the heart .",
        "I 've changed places and I do n't know which one is the best for me .",
        "I am an easy going person with a lot of empathy for children .",
        "Just paste your text here and check the ' Check Text ' button .",
        "I click the colored phrases for details on potential errors .",
        "Or use this text to see a few of the problems that LanguageTool can detect .",
        "I am an easy going person with a lot of empathy for children ."
    )
    val textSimpleMistakes = listOf("It are bad .",
        "It are even more interesting .",
        "What I sure about is the fact that I am going to need an official certificate in order to prove that I 've studied the language .",
        "I am easy going person with a lot of empathy for children .",
        "I study English because I love it and I 'd like to speak very fluently and watch a movie without a subtitles .",
        "And I likes cake too .",
        "It are bad things to do and I think it is a good thing s to do it and")

    val textComplexMistakes = listOf(
        "I have travelled around most of European countries and I be able to understanding , talking and writin highly three languages: Spanish, English and French.",
        "I 'll hopes it does not involve inconvenience to you .",
        "It is give to the body the energy as well as regular work of the heart .")
    for (str in textComplexMistakes){
        val answer = gecCorrector.evaluateCorrectionsInFewIterations(str)
        println(answer)
    }
    println("DONE")
//    val answers = gecCorrector.evaluateCorrectionsListInFewIterations(testStrings)
//    println(answers)
//    println(answers.size)
}
