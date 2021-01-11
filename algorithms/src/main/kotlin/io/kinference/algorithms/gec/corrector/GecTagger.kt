package io.kinference.algorithms.gec.corrector

import io.kinference.algorithms.gec.preprocessing.TransformersTextprocessor
import io.kinference.algorithms.gec.preprocessing.Vocabulary
import io.kinference.algorithms.gec.preprocessing.tagKeep
import io.kinference.algorithms.gec.utils.TagSentObject
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.forEach
import io.kinference.ndarray.extensions.concatenate
import io.kinference.ndarray.extensions.gather
import io.kinference.operators.activations.Softmax

data class GecTaggerResult(val tagsIds: IntNDArray,
                           val porbsTags: FloatNDArray,
                           val maxIncorrectProb: FloatNDArray)

class GecTagger(val model: Seq2Logits,
                val textProcessor: TransformersTextprocessor,
                val labelsVocabulary: Vocabulary,
                val dTagsVocabulary: Vocabulary,
                val minCorrectionProb: Double,
                val minErrorProb: Double,
                val confidence: Double, )
{
    fun correctList(sentences: List<GecTaggerFeatures>, batchSize: Int = 20): List<TagSentObject>{
        val dataset = GecTaggerFeaturesData(sentences)
        val loader = GecTaggerFeaturesDataLoader(dataset = dataset, batchSize = 20, padId = textProcessor.tokenizer.padId)
        val tagsObjects = ArrayList<TagSentObject>()
        for ((batchIdx, batch) in loader.withIndex()){

            val sentIds = batch.sentIds
            val offsetBatch = batch.offsets
            val sentsBatch = batch.sentences
            val lens = batch.lens

            val innerSentsSize = sentsBatch[0].size
            val innerOffsetSize = offsetBatch[0].size
            val tensorSent = LongNDArray(shape = IntArray(size = 2, init = {i: Int ->  if (i==0) sentsBatch.size else innerSentsSize}),
                init = {i: Int -> sentsBatch[i / innerSentsSize][i % innerSentsSize].toLong() })

            val tensorOffset = LongNDArray(shape = IntArray(size = 2, init = {i: Int ->  if (i==0) offsetBatch.size else innerOffsetSize}),
                init = {i: Int -> offsetBatch[i / innerOffsetSize][i % innerOffsetSize].toLong() })

            val attentionMask = tensorSent.map(object : LongMap {
                override fun apply(value: Long): Long {
                    return if (value != textProcessor.padId.toLong())
                        1
                    else
                        0
                }
            }) as LongNDArray

            val result: GecTaggerResult = predictTags(tensorSent, attentionMask, tensorOffset)
            for (idx in 0 until result.tagsIds.shape[0]){
                val tagsSentIds = ArrayList<Int>()
                result.tagsIds.array.pointer(startIndex = idx*innerOffsetSize).forEach(count = lens[idx], action = { tagsSentIds.add(it) } )

                val probsSentTags = ArrayList<Float>()
                result.porbsTags.array.pointer(startIndex = idx*innerOffsetSize).forEach(count = lens[idx], action = { probsSentTags.add(it) })

                val maxSentIncorrectDTags = result.maxIncorrectProb
                assert(tagsSentIds.size == sentences[batchSize * batchIdx + idx].tokSent.size)

                val tags = decodeTags(tagsSentIds, probsSentTags, maxSentIncorrectDTags.array.pointer().get())

                tagsObjects.add(TagSentObject(sentId = sentIds[idx], tags = tags, tokens = sentences[batchSize * batchIdx + idx].tokSent))
            }
        }
        return tagsObjects
    }

    fun correct(sentence: GecTaggerFeatures): TagSentObject{
        val encs = sentence.flatEncSent
        val tokens = sentence.tokSent
        val offset = sentence.offset

        val innerSize = encs.size
        val tensorSent = LongNDArray(shape = IntArray(size = 2, init = { i: Int -> if (i==0) 1 else innerSize}), init = { i: Int -> encs[i].toLong()})
        val tensorOffset = LongNDArray(shape = IntArray(size = 1, init = { offset.size }), init = {i: Int -> offset[i].toLong()})

        val attentionMask = tensorSent.map(object : LongMap {
            override fun apply(value: Long): Long {
                return if (value != textProcessor.padId.toLong())
                    1
                else
                    0
            }
        }) as LongNDArray

        val result: GecTaggerResult = predictTags(tensorSent, attentionMask, tensorOffset)

        val tagsSentIds = ArrayList<Int>()
        result.tagsIds.array.pointer(startIndex = 0).forEach(count = result.tagsIds.shape[1], action = { tagsSentIds.add(it) } )

        val probsSentTags = ArrayList<Float>()
        result.porbsTags.array.pointer(startIndex = 0).forEach(count = result.porbsTags.shape[1], action = { probsSentTags.add(it) })

        val tags = decodeTags(tagsIds = tagsSentIds, porbsTags = probsSentTags, maxIncorrectProb = result.maxIncorrectProb.array.pointer(0).get())

        return TagSentObject(sentId = sentence.sentId, tags = tags, tokens = tokens)

    }

    fun badArgmax(tensor: FloatNDArray): IntNDArray{
        val argMaxTensor = IntNDArray(shape = IntArray(size = 2, init = {i: Int -> if (i==0) tensor.shape[0] else tensor.shape[1]}), init = { 0 })

        for (batchIdx in 0 until tensor.shape[0]){
            for (seqIdx in 0 until tensor.shape[1]){
                val tmpVector= ArrayList<Float>()
                tensor.array.pointer(startIndex = (seqIdx + batchIdx * tensor.shape[1]) * tensor.shape[2]).forEach(count = tensor.shape[2], action = {tmpVector.add(it)})
                tmpVector.withIndex().maxByOrNull { it.value }?.index?.let { argMaxTensor.array.pointer(startIndex = batchIdx*tensor.shape[1] + seqIdx).set(it) }
            }
        }
        return argMaxTensor
    }

    fun predictTags(sents: LongNDArray, attentionMask: LongNDArray,  offset: LongNDArray): GecTaggerResult{
        val logitsResult = model(sents, attentionMask)
        var logitsTags = logitsResult.logitsTag
        var logitsDTags = logitsResult.logitsDTags

        val batchSize = logitsTags.shape[0]
        val seqSize = logitsTags.shape[1]
        val hiddenSize = logitsTags.shape[2]

        if (offset.shape.size == 1){
            logitsTags = logitsTags.gather(indices = offset, axis = 1) as FloatNDArray
            logitsDTags = logitsDTags.gather(indices = offset, axis = 1) as FloatNDArray
        }
        else{
            val logitsTagsPerBatch = ArrayList<FloatNDArray>()
            val logitsDTagsPerBatch = ArrayList<FloatNDArray>()
            for (batchIdx in 0 until batchSize){
                var logitsTagsSlice = logitsTags.slice(starts = listOf(batchIdx, 0, 0).toIntArray(), ends = listOf(batchIdx + 1, logitsTags.shape[1], logitsTags.shape[2]).toIntArray(), steps = listOf(1, 1, 1).toIntArray()) as FloatNDArray
                var logitsDTagsSlice = logitsDTags.slice(starts = listOf(batchIdx, 0, 0).toIntArray(), ends = listOf(batchIdx + 1, logitsDTags.shape[1], logitsDTags.shape[2]).toIntArray(), steps = listOf(1, 1, 1).toIntArray()) as FloatNDArray
                val offsetSlice = offset.slice(starts = listOf(batchIdx, 0, 0).toIntArray(), ends = listOf(batchIdx + 1, offset.shape[1]).toIntArray(), steps = listOf(1, 1).toIntArray()).reshape(shape = listOf(offset.shape[1]).toIntArray()) as LongNDArray

//                var offsetMask = offsetSlice.map(object : LongMap {
//                    override fun apply(value: Long): Long {
//                        return if (value != 0L)
//                            1
//                        else
//                            0
//                    }
//                }) as LongNDArray


                logitsTagsSlice = logitsTagsSlice.gather(indices = offsetSlice, axis = 1) as FloatNDArray
                logitsDTagsSlice = logitsDTagsSlice.gather(indices = offsetSlice, axis = 1) as FloatNDArray

//                logitsTagsSlice = (logitsTagsSlice * offsetMask) as FloatNDArray
//                logitsDTagsSlice = (logitsDTagsSlice * offsetMask) as FloatNDArray
                logitsTagsPerBatch.add(logitsTagsSlice)
                logitsDTagsPerBatch.add(logitsDTagsSlice)
            }
            logitsTags = logitsTagsPerBatch.concatenate(axis = 0) as FloatNDArray
            logitsDTags = logitsDTagsPerBatch.concatenate(axis = 0) as FloatNDArray
        }

        val probsTags = Softmax.softmax(logitsTags, axis = logitsTags.shape.lastIndex) as FloatNDArray
        val probsDTags = Softmax.softmax(logitsDTags, axis = logitsDTags.shape.lastIndex) as FloatNDArray

        if (confidence > 0.0){
            for (batchIdx in 0 until batchSize){
                for (seqIdx in 0 until seqSize){
                    val value = probsTags.array.pointer(startIndex = batchIdx*seqSize + seqIdx + 3).get() + confidence
                    probsTags.array.pointer(startIndex = batchIdx*seqSize + seqIdx + 3).set(value = (value as Float))
                }
            }
        }

        val tags = badArgmax(probsTags)
        val probsBestTags = FloatNDArray(shape = IntArray(size = 2, init = {i: Int -> if (i==0) probsTags.shape[0] else probsTags.shape[1]}), init = { 0.0f })

        for (batchIdx in 0 until batchSize){
            for (seqIdx in 0 until probsTags.shape[1]){
                val value = probsTags.array.pointer(startIndex = (batchIdx * probsTags.shape[1] + seqIdx ) * probsTags.shape[2] + tags.array.pointer(batchIdx*tags.shape[1] + seqIdx).get()).get()
                probsBestTags.array.pointer(batchIdx*probsTags.shape[1] + seqIdx).set(value = value)
            }
        }

        val probsIncorrect = FloatNDArray(shape = IntArray(size = 1, init = { batchSize }), init = { 0.0f })

        for (batchIdx in 0 until batchSize){
            var sentenceIncorrProbs = ArrayList<Float>()

            for (seqIdx in 0 until probsDTags.shape[1]){
                sentenceIncorrProbs.add(probsDTags.array.pointer(startIndex = batchIdx*probsDTags.shape[1] + seqIdx * probsDTags.shape[2] + 3).get())
            }
            sentenceIncorrProbs!!.max()?.let { probsIncorrect.array.pointer(startIndex = batchIdx).set(value = it) }
        }
        assert(tags.shape.lastIndex == probsBestTags.shape.lastIndex)
        assert(tags.shape[0] == probsIncorrect.shape[0])

        return GecTaggerResult(tagsIds = tags, porbsTags = probsBestTags, maxIncorrectProb = probsIncorrect)
    }

    fun decodeTags(tagsIds: List<Int>, porbsTags: ArrayList<Float>, maxIncorrectProb: Float): List<String>{
        if (maxIncorrectProb < minErrorProb){
            return tagsIds.map { tagKeep.value }
        }

        return porbsTags.mapIndexed { index, prob -> if (prob > minCorrectionProb) labelsVocabulary.getTokenByIndex(tagsIds[index]) else tagKeep.value }
    }
}
