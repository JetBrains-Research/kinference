package io.kinference.algorithms.gec.model

import io.kinference.algorithms.gec.ConfigLoader
import io.kinference.algorithms.gec.corrector.GECCorrector
import io.kinference.algorithms.gec.postprocessing.GecCorrectionPostprocessor
import io.kinference.algorithms.gec.postprocessing.GecPostprocessor
import io.kinference.algorithms.gec.preprocessing.GecCorrectionPreprocessor
import io.kinference.algorithms.gec.preprocessing.GecPreprocessor
import io.kinference.algorithms.gec.utils.TextCorrection
import org.junit.jupiter.api.*

class CorrectorTest {

    private val config = ConfigLoader.v2

    private val preprocessor: GecPreprocessor = GecCorrectionPreprocessor(encoder = config.encoder, useStartToken = true)
    private val postprocessor: GecPostprocessor = GecCorrectionPostprocessor()

    private val gecCorrector = GECCorrector(
        config.model,
        config.encoder,
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
        listOf(TextCorrection(errorRange = IntRange(3, 5), underlineRange = IntRange(3, 5), replacement = "is", message = "Grammatical error")),
        listOf(TextCorrection(errorRange = IntRange(3, 5), underlineRange = IntRange(3, 5), replacement = "is", message = "Grammatical error")),
        listOf(TextCorrection(errorRange = IntRange(5, 5), underlineRange = IntRange(5, 5), replacement = "I am", message = "Grammatical error")),
        listOf(TextCorrection(errorRange = IntRange(2, 14), underlineRange = IntRange(2, 14), replacement = "am an easy", message = "Complex error")),
        listOf(TextCorrection(errorRange = IntRange(97, 98), underlineRange = IntRange(97, 97), replacement = "", message = "Grammatical error")),
        listOf(TextCorrection(errorRange = IntRange(6, 10), underlineRange = IntRange(6, 10), replacement = "like", message = "Incorrect verb form")),
        listOf(
            TextCorrection(errorRange = IntRange(0, 1), underlineRange = IntRange(0, 1), replacement = "There", message = "Grammatical error"),
            TextCorrection(errorRange = IntRange(55, 56), underlineRange = IntRange(55, 55), replacement = "", message = "Grammatical error")
        )
    )

    private val textComplexMistakes = listOf(
        "I 've change the place and I do n't knows which one is the best for me .",
        "As I 'm new here , I 'm lost and do n't know where is to my hotel .",
        "You do n't need to worry about your writing skills any more , improving you 're text has never be more easier !",
        "It are an friend .",
        "I have travelled around most of European countries and I be able to understanding , talking and writin highly three languages: Spanish, English and French.",
        "I 'll hopes it does not involve inconvenience to you .",
        "It is give to the body the energy as well as regular work of the heart ."
    )

    private val answersComplexMistakes = listOf(
        listOf(
            TextCorrection(errorRange = IntRange(6, 11), underlineRange = IntRange(6, 11), replacement = "changed", message = "Incorrect verb form"),
            TextCorrection(errorRange = IntRange(36, 40), underlineRange = IntRange(36, 40), replacement = "know", message = "Incorrect verb form")
        ),
        emptyList(),
        listOf(TextCorrection(errorRange = IntRange(95, 101), underlineRange = IntRange(95, 101), replacement = "been", message = "Complex error")),
        listOf(TextCorrection(errorRange = IntRange(3, 15), underlineRange = IntRange(3, 15), replacement = "is friend", message = "Complex error")),
        listOf(
            TextCorrection(errorRange = IntRange(29, 31), underlineRange = IntRange(29, 30), replacement = "", message = "Grammatical error"),
            TextCorrection(errorRange = IntRange(55, 55), underlineRange = IntRange(55, 55), replacement = "I will", message = "Grammatical error"),
            TextCorrection(
                errorRange = IntRange(68, 108),
                underlineRange = IntRange(68, 108),
                replacement = "understand , talk and write",
                message = "Complex error"
            )
        ),
        listOf(TextCorrection(errorRange = IntRange(6, 10), underlineRange = IntRange(6, 10), replacement = "hope", message = "Incorrect verb form")),
        listOf(TextCorrection(errorRange = IntRange(6, 9), underlineRange = IntRange(6, 9), replacement = "given", message = "Incorrect verb form"))
    )

    private val textNoMistakes = listOf(
        "I 've changed places and I do n't know which one is the best for me .",
        "I am an easy going person with a lot of empathy for children .",
        "Just paste your text here and check the ' Check Text ' button .",
        "I click the colored phrases for details on potential errors .",
        "Or use this text to see a few of the problems that LanguageTool can detect .",
        "I am an easy going person with a lot of empathy for children .",
        "Or use this text to see a few of the problems that LanguageTool can detect .",
        "What I am sure about is the fact that I am going to need an official certificate in order to prove that I 've studied the language .",
        "I 've seen your advertisements for jobs on the internet and I 'm writing to apply for a summer job as an instructor and keeper of children in your camp ."
    )

    private val answersEmptyMistates = listOf(
        emptyList(),
        emptyList(),
        emptyList(),
        emptyList(),
        emptyList(),
        emptyList(),
        emptyList(),
        emptyList(),
        emptyList<TextCorrection>()
    )

    @Test
    @Tag("heavy")
    fun testSimpleMistakes() {
        val oneSimpleMistakes = ArrayList<List<TextCorrection>>()

        for (text in textSimpleMistakes) {
            oneSimpleMistakes.add(gecCorrector.evaluateCorrectionsInFewIterations(text))
        }

        val batchSimpleMistakes = gecCorrector.evaluateCorrectionsListInFewIterations(textSimpleMistakes)

        Assertions.assertEquals(answersSimpleMistakes.size, oneSimpleMistakes.size)
        Assertions.assertEquals(answersSimpleMistakes.size, batchSimpleMistakes.size)
        for (idx in textSimpleMistakes.indices) {
            Assertions.assertEquals(answersSimpleMistakes[idx], oneSimpleMistakes[idx])
            Assertions.assertEquals(answersSimpleMistakes[idx], batchSimpleMistakes[idx])
        }
    }

    @Test
    @Tag("heavy")
    fun testComplexMistakes() {
        val oneComplexMistakes = ArrayList<List<TextCorrection>>()

        for (text in textComplexMistakes) {
            oneComplexMistakes.add(gecCorrector.evaluateCorrectionsInFewIterations(text))
        }

        val batchComplexMistakes = gecCorrector.evaluateCorrectionsListInFewIterations(textComplexMistakes)

        assert(oneComplexMistakes.size == answersComplexMistakes.size)
        assert(answersComplexMistakes.size == batchComplexMistakes.size)
        for (idx in textComplexMistakes.indices) {
            Assertions.assertEquals(oneComplexMistakes[idx], answersComplexMistakes[idx])
            Assertions.assertEquals(batchComplexMistakes[idx], answersComplexMistakes[idx])
        }
    }

    @Test
    @Tag("heavy")
    fun testEmptyMistakes() {
        val oneEmptyMistakes = ArrayList<List<TextCorrection>>()

        for (text in textNoMistakes) {
            oneEmptyMistakes.add(gecCorrector.evaluateCorrectionsInFewIterations(text))
        }

        val batchEmptyMistakes = gecCorrector.evaluateCorrectionsListInFewIterations(textNoMistakes)

        assert(oneEmptyMistakes.size == answersEmptyMistates.size)
        assert(answersEmptyMistates.size == batchEmptyMistakes.size)
        for (idx in textComplexMistakes.indices) {
            Assertions.assertEquals(oneEmptyMistakes[idx], answersEmptyMistates[idx])
            Assertions.assertEquals(batchEmptyMistakes[idx], answersEmptyMistates[idx])
        }
    }
}
