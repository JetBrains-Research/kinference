package io.kinference.algorithms.gec.corrector

import io.kinference.algorithms.gec.GECTag
import io.kinference.algorithms.gec.encoder.PreTrainedTextEncoder
import io.kinference.algorithms.gec.preprocessing.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.forEach
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.extensions.gather
import io.kinference.operators.activations.Softmax
import io.kinference.primitives.types.DataType

/**
 * Class for generation tags from sentence
 * @param model Seq2Logits class which can predict tags for each token in sequence
 * @param encoder TransformersTextprocessor class for processing sequences
 * @param labelsVocabulary label vocabulary
 * @param dTagsVocabulary detection tags vocabulary
 * @param minCorrectionProb minimum probability for correction
 * @param minErrorProb minimum probability for error
 * @param confidence bias for not-changing strategy
 */
class GECTagger(
    val model: Seq2Logits,
    val encoder: PreTrainedTextEncoder,
    val labelsVocabulary: TokenVocabulary,
    val dTagsVocabulary: TokenVocabulary,
    val minCorrectionProb: Double,
    val minErrorProb: Double,
    val confidence: Double,
) {
    data class TagSentObject(val sentId: Int, val tags: List<String>, val tokens: List<String>)


    /**
     * result class from GecTagger for realization corrections
     */
    data class Result(val tagsIds: IntNDArray, val probsTags: FloatNDArray, val maxIncorrectProb: FloatNDArray)

    /**
     * tags generation from list of features
     */
    fun correctList(sentences: List<GECTaggerFeatures>, batchSize: Int = 20): List<TagSentObject> {
        val loader = GECTaggerFeatures.DataLoader(dataset = sentences, batchSize = 20, padId = encoder.padId)

        val tagsObjects = ArrayList<TagSentObject>()
        for ((batchIdx, batch) in loader.withIndex()) {
            val (sentIds, sentsBatch, offsetBatch, lens) = batch

            val innerSentsSize = sentsBatch[0].size
            val innerOffsetSize = offsetBatch[0].size

            val tensorSent = LongNDArray(shape = IntArray(size = 2, init = { i: Int -> if (i == 0) sentsBatch.size else innerSentsSize }),
                init = { i: Int -> sentsBatch[i / innerSentsSize][i % innerSentsSize].toLong() })

            val tensorOffset = LongNDArray(shape = IntArray(size = 2, init = { i: Int -> if (i == 0) offsetBatch.size else innerOffsetSize }),
                init = { i: Int -> offsetBatch[i / innerOffsetSize][i % innerOffsetSize].toLong() })

            val attentionMask = tensorSent.map(object : LongMap {
                override fun apply(value: Long): Long = if (value != encoder.padId.toLong()) 1 else 0}
            ) as LongNDArray

            val result: Result = predictTags(tensorSent, attentionMask, tensorOffset)
            for (idx in 0 until result.tagsIds.shape[0]) {
                val tagsSentIds = ArrayList<Int>()
                result.tagsIds.array.pointer(startIndex = idx * innerOffsetSize).forEach(count = lens[idx], action = { tagsSentIds.add(it) })

                val probsSentTags = ArrayList<Float>()
                result.probsTags.array.pointer(startIndex = idx * innerOffsetSize).forEach(count = lens[idx], action = { probsSentTags.add(it) })

                val maxSentIncorrectDTags = result.maxIncorrectProb
                require(tagsSentIds.size == sentences[batchSize * batchIdx + idx].tokSent.size)

                val tags = decodeTags(tagsSentIds, probsSentTags, maxSentIncorrectDTags.array.pointer().get())

                tagsObjects.add(TagSentObject(sentId = sentIds[idx], tags = tags, tokens = sentences[batchSize * batchIdx + idx].tokSent))
            }
        }
        return tagsObjects
    }

    /**
     * tags generation from of sentence feature
     */
    fun correct(sentence: GECTaggerFeatures): TagSentObject {
        val encs = sentence.flatEncSent
        val tokens = sentence.tokSent
        val offset = sentence.offset

        val innerSize = encs.size
        val tensorSent = LongNDArray(shape = IntArray(size = 2, init = { i: Int -> if (i == 0) 1 else innerSize }), init = { i: Int -> encs[i].toLong() })
        val tensorOffset = LongNDArray(shape = IntArray(size = 1, init = { offset.size }), init = { i: Int -> offset[i].toLong() })

        val attentionMask = tensorSent.map(object : LongMap {
            override fun apply(value: Long): Long = if (value != encoder.padId.toLong()) 1 else 0
        }) as LongNDArray

        val result: Result = predictTags(tensorSent, attentionMask, tensorOffset)

        val tagsSentIds = ArrayList<Int>()
        result.tagsIds.array.pointer(startIndex = 0).forEach(count = result.tagsIds.shape[1], action = { tagsSentIds.add(it) })

        val probsSentTags = ArrayList<Float>()
        result.probsTags.array.pointer(startIndex = 0).forEach(count = result.probsTags.shape[1], action = { probsSentTags.add(it) })

        val tags = decodeTags(tagsIds = tagsSentIds, probsTags = probsSentTags, maxIncorrectProb = result.maxIncorrectProb.array.pointer(0).get())

        return TagSentObject(sentId = sentence.sentId, tags = tags, tokens = tokens)

    }

    private fun badArgmax(tensor: FloatNDArray): IntNDArray {
        val argMaxTensor = IntNDArray(shape = IntArray(size = 2, init = { i: Int -> if (i == 0) tensor.shape[0] else tensor.shape[1] }), init = { 0 })

        for (batchIdx in 0 until tensor.shape[0]) {
            for (seqIdx in 0 until tensor.shape[1]) {
                val tmpVector = ArrayList<Float>()
                tensor.array.pointer(startIndex = (seqIdx + batchIdx * tensor.shape[1]) * tensor.shape[2]).forEach(count = tensor.shape[2], action = { tmpVector.add(it) })
                tmpVector.withIndex().maxByOrNull { it.value }?.index?.let { argMaxTensor.array.pointer(startIndex = batchIdx * tensor.shape[1] + seqIdx).set(it) }
            }
        }
        return argMaxTensor
    }

    private fun predictTags(sents: LongNDArray, attentionMask: LongNDArray, offset: LongNDArray): Result {
        val logitsResult = model(sents, attentionMask)
        var logitsTags = logitsResult.logitsTag
        var logitsDTags = logitsResult.logitsDTags

        val batchSize = logitsTags.shape[0]
        val seqSize = logitsTags.shape[1]
        val hiddenSize = logitsTags.shape[2]

        val actualOffset = if (offset.shape.size == 1) offset.reshapeView(intArrayOf(1, offset.shape[0])) else offset

        val logitsTagsOutput = allocateNDArray(DataType.FLOAT, intArrayOf(batchSize, actualOffset.shape[1], hiddenSize))
        val logitsDTagsOutput = allocateNDArray(DataType.FLOAT, intArrayOf(batchSize, actualOffset.shape[1], logitsDTags.shape.last()))

        for (batchIdx in 0 until batchSize) {
            val logitsTagsSlice = logitsTags.view(batchIdx)
            val logitsDTagsSlice = logitsDTags.view(batchIdx)
            val offsetSlice = actualOffset.view(batchIdx)

            val logitsTagsOutputSlice = logitsTagsOutput.viewMutable(batchIdx)
            val logitsDTagsOutputSlice = logitsDTagsOutput.viewMutable(batchIdx)

            logitsTagsSlice.gather(offsetSlice, 0, logitsTagsOutputSlice)
            logitsDTagsSlice.gather(offsetSlice, 0, logitsDTagsOutputSlice)
        }

        logitsTags = logitsTagsOutput as FloatNDArray
        logitsDTags = logitsDTagsOutput as FloatNDArray

        val probsTags = Softmax.softmax(logitsTags, axis = logitsTags.shape.lastIndex) as FloatNDArray
        val probsDTags = Softmax.softmax(logitsDTags, axis = logitsDTags.shape.lastIndex) as FloatNDArray

        if (confidence > 0.0) {
            for (batchIdx in 0 until batchSize) {
                for (seqIdx in 0 until seqSize) {
                    val value = probsTags.array.pointer(startIndex = batchIdx * seqSize + seqIdx + 3).get() + confidence
                    probsTags.array.pointer(startIndex = batchIdx * seqSize + seqIdx + 3).set(value = value.toFloat())
                }
            }
        }

        val tags = badArgmax(probsTags)
        val probsBestTags = FloatNDArray(shape = IntArray(size = 2, init = { i: Int -> if (i == 0) probsTags.shape[0] else probsTags.shape[1] }), init = { 0.0f })

        for (batchIdx in 0 until batchSize) {
            for (seqIdx in 0 until probsTags.shape[1]) {
                val value = probsTags.array.pointer(startIndex = (batchIdx * probsTags.shape[1] + seqIdx) * probsTags.shape[2]
                    + tags.array.pointer(batchIdx * tags.shape[1] + seqIdx).get()).get()
                probsBestTags.array.pointer(batchIdx * probsTags.shape[1] + seqIdx).set(value = value)
            }
        }

        val probsIncorrect = FloatNDArray(shape = IntArray(size = 1, init = { batchSize }), init = { 0.0f })

        for (batchIdx in 0 until batchSize) {
            val sentenceIncorrectProbs = ArrayList<Float>()

            for (seqIdx in 0 until probsDTags.shape[1]) {
                sentenceIncorrectProbs.add(probsDTags.array.pointer(startIndex = batchIdx * probsDTags.shape[1] + seqIdx * probsDTags.shape[2] + 3).get())
            }
            sentenceIncorrectProbs.maxOrNull()?.let { probsIncorrect.array.pointer(startIndex = batchIdx).set(value = it) }
        }
        require(tags.shape.lastIndex == probsBestTags.shape.lastIndex)
        require(tags.shape[0] == probsIncorrect.shape[0])

        return Result(tagsIds = tags, probsTags = probsBestTags, maxIncorrectProb = probsIncorrect)
    }

    private fun decodeTags(tagsIds: List<Int>, probsTags: ArrayList<Float>, maxIncorrectProb: Float): List<String> {
        if (maxIncorrectProb < minErrorProb) {
            return tagsIds.map { GECTag.KEEP.value }
        }

        return probsTags.mapIndexed { index, prob -> if (prob > minCorrectionProb) labelsVocabulary.getTokenByIndex(tagsIds[index]) else GECTag.KEEP.value }
    }
}
