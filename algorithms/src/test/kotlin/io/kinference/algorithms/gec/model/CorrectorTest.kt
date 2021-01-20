package io.kinference.algorithms.gec.model

import io.kinference.algorithms.gec.ConfigLoader
import io.kinference.algorithms.gec.corrector.GECCorrector
import io.kinference.algorithms.gec.postprocessing.GecCorrectionPostprocessor
import io.kinference.algorithms.gec.postprocessing.GecPostprocessor
import io.kinference.algorithms.gec.preprocessing.GecCorrectionPreprocessor
import io.kinference.algorithms.gec.preprocessing.GecPreprocessor
import io.kinference.algorithms.gec.utils.TextCorrection
import org.junit.jupiter.api.*

fun getFromResources(path: String): String = object {}.javaClass.getResource(path)!!.path!!

class CorrectorTest {

    private val config = ConfigLoader.v2

    private val preprocessor: GecPreprocessor = GecCorrectionPreprocessor(textProcessor = config.textProcessor, useStartToken = true, useForEval = false)
    private val postprocessor: GecPostprocessor = GecCorrectionPostprocessor()

    private val gecCorrector = GECCorrector(
        config.model,
        config.textProcessor,
        config.labelsVocab,
        config.dTagsVocab,
        config.verbsVocab,
        preprocessor,
        postprocessor,
        iterations = 5
    )

    private val textSimpleMistakes = listOf(
        "It are bad .",
        "It are even more interesting .",
        "What I sure about is the fact that I am going to need an official certificate in order to prove that I 've studied the language .",
        "I am easy going person with a lot of empathy for children .",
        "I study English because I love it and I 'd like to speak very fluently and watch a movie without a subtitles .",
        "And I likes cake too .",
        "It are bad things to do and I think it is a good thing s to do it and"
    )

    private val answersSimpleMistakes = listOf(
        listOf(TextCorrection(errorRange = Pair(3, 6), underlineRange = Pair(3, 6), replacement = "is", massage = "Grammatical error")),
                                               listOf(TextCorrection(errorRange = Pair(3, 6), underlineRange = Pair(3, 6), replacement = "is", massage = "Grammatical error")),
                                               listOf(TextCorrection(errorRange = Pair(5, 6), underlineRange = Pair(5, 6), replacement = "I am", massage = "Grammatical error")),
                                               listOf(TextCorrection(errorRange = Pair(2, 15), underlineRange = Pair(2, 15), replacement = "am an easy", massage = "Complex error")),
                                               listOf(TextCorrection(errorRange = Pair(97, 99), underlineRange = Pair(97, 98), replacement = "", massage = "Grammatical error")),
                                               listOf(TextCorrection(errorRange = Pair(6, 11), underlineRange = Pair(6, 11), replacement = "like", massage = "Incorrect verb form")),
                                               listOf(TextCorrection(errorRange = Pair(0, 2), underlineRange = Pair(0, 2), replacement = "There", massage = "Grammatical error"), TextCorrection(errorRange = Pair(55, 57), underlineRange = Pair(55, 56), replacement = "", massage = "Grammatical error")))

    private val textComplexMistakes = listOf("I 've change the place and I do n't knows which one is the best for me .",
                                             "As I 'm new here , I 'm lost and do n't know where is to my hotel .",
                                             "You do n't need to worry about your writing skills any more , improving you 're text has never be more easier !",
                                             "It are an friend .",
                                             "I have travelled around most of European countries and I be able to understanding , talking and writin highly three languages: Spanish, English and French.",
                                             "I 'll hopes it does not involve inconvenience to you .",
                                             "It is give to the body the energy as well as regular work of the heart .")

    private val answersComplexMistakes = listOf(listOf(TextCorrection(errorRange = Pair(6, 12), underlineRange = Pair(6, 12), replacement = "changed", massage = "Incorrect verb form"), TextCorrection(errorRange=Pair(36, 41), underlineRange=Pair(36, 41), replacement="know", massage="Incorrect verb form")),
                                                emptyList(),
                                                listOf(TextCorrection(errorRange= Pair(95, 102), underlineRange=Pair(95, 102), replacement="been", massage="Complex error")),
                                                listOf(TextCorrection(errorRange=Pair(3, 16), underlineRange=Pair(3, 16), replacement="is friend", massage="Complex error")),
                                                listOf(TextCorrection(errorRange=Pair(29, 32), underlineRange=Pair(29, 31), replacement="", massage="Grammatical error"), TextCorrection(errorRange=Pair(55, 56), underlineRange=Pair(55, 56), replacement="I will", massage="Grammatical error"), TextCorrection(errorRange=Pair(68, 109), underlineRange=Pair(68, 109), replacement="understand , talk and write", massage="Complex error")),
                                                listOf(TextCorrection(errorRange=Pair(6, 11), underlineRange=Pair(6, 11), replacement="hope", massage="Incorrect verb form")),
                                                listOf(TextCorrection(errorRange=Pair(6, 10), underlineRange=Pair(6, 10), replacement="given", massage="Incorrect verb form"))
                                                )

    private val textNoMistakes = listOf("I 've changed places and I do n't know which one is the best for me .",
                                        "I am an easy going person with a lot of empathy for children .",
                                        "Just paste your text here and check the ' Check Text ' button .",
                                        "I click the colored phrases for details on potential errors .",
                                        "Or use this text to see a few of the problems that LanguageTool can detect .",
                                        "I am an easy going person with a lot of empathy for children .",
                                        "Or use this text to see a few of the problems that LanguageTool can detect .",
                                        "What I am sure about is the fact that I am going to need an official certificate in order to prove that I 've studied the language .",
                                        "I 've seen your advertisements for jobs on the internet and I 'm writing to apply for a summer job as an instructor and keeper of children in your camp .")

    private val answersEmptyMistates = listOf(emptyList(),
                                              emptyList(),
                                              emptyList(),
                                              emptyList(),
                                              emptyList(),
                                              emptyList(),
                                              emptyList(),
                                              emptyList(),
                                              emptyList<TextCorrection>())

    @Test
    @Tag("heavy")
    fun testSimpleMistak(){
        val oneSimpleMistakes = ArrayList<List<TextCorrection>>()

        for (text in textSimpleMistakes){
            oneSimpleMistakes.add(gecCorrector.evaluateCorrectionsInFewIterations(text))
        }

        val batchSimpleMistakes = gecCorrector.evaluateCorrectionsListInFewIterations(textSimpleMistakes)

        assert(oneSimpleMistakes.size == answersSimpleMistakes.size)
        assert(answersSimpleMistakes.size == batchSimpleMistakes.size)
        for (idx in textSimpleMistakes.indices){
            Assertions.assertEquals(oneSimpleMistakes[idx], answersSimpleMistakes[idx])
            Assertions.assertEquals(batchSimpleMistakes[idx], answersSimpleMistakes[idx])
        }
    }

    @Test
    @Tag("heavy")
    fun testComplexMistakes(){
        val oneComplexMistakes = ArrayList<List<TextCorrection>>()

        for (text in textComplexMistakes){
            oneComplexMistakes.add(gecCorrector.evaluateCorrectionsInFewIterations(text))
        }

        val batchComplexMistakes = gecCorrector.evaluateCorrectionsListInFewIterations(textComplexMistakes)

        assert(oneComplexMistakes.size == answersComplexMistakes.size)
        assert(answersComplexMistakes.size == batchComplexMistakes.size)
        for (idx in textComplexMistakes.indices){
            Assertions.assertEquals(oneComplexMistakes[idx], answersComplexMistakes[idx])
            Assertions.assertEquals(batchComplexMistakes[idx], answersComplexMistakes[idx])
        }
    }

    @Test
    @Tag("heavy")
    fun testEmptyMistakes(){
        val oneEmptyMistakes = ArrayList<List<TextCorrection>>()

        for (text in textNoMistakes){
            oneEmptyMistakes.add(gecCorrector.evaluateCorrectionsInFewIterations(text))
        }

        val batchEmptyMistakes = gecCorrector.evaluateCorrectionsListInFewIterations(textNoMistakes)

        assert(oneEmptyMistakes.size == answersEmptyMistates.size)
        assert(answersEmptyMistates.size == batchEmptyMistakes.size)
        for (idx in textComplexMistakes.indices){
            Assertions.assertEquals(oneEmptyMistakes[idx], answersEmptyMistates[idx])
            Assertions.assertEquals(batchEmptyMistakes[idx], answersEmptyMistates[idx])
        }
    }
}
