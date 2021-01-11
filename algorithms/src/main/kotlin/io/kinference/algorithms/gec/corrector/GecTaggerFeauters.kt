package io.kinference.algorithms.gec.corrector

import kotlin.math.min


data class GecTaggerFeatures(val sent: String,
                             val sentId: Int,
                             val tokSent: List<String>,
                             val encTokSent: List<List<Int>>,
                             val flatEncSent: List<Int>,
                             val offset: List<Int>)

data class GecTaggerFeaturesOutput(val sentIds: List<Int>,
                                   val sentences: List<List<Int>>,
                                   val offsets: List<List<Int>>,
                                   val lens: List<Int>)

data class GecTaggerIteratorOut(val sentId: Int,
                                val sentence: List<Int>,
                                val offset: List<Int>,
                                val len: Int)

class GecTaggerFeaturesData(val data: List<GecTaggerFeatures>){
    operator fun get(index: Int): GecTaggerIteratorOut{
        val sentence = data[index]
        return GecTaggerIteratorOut(
            sentence.sentId,
            sentence.flatEncSent,
            sentence.offset,
            sentence.tokSent.size)
    }

    fun size(): Int{
        return data.size
    }
}

class GecTaggerFeaturesDataLoader(
    val dataset: GecTaggerFeaturesData,
    val batchSize: Int, val padId: Int) : Iterable<GecTaggerFeaturesOutput>{

    override fun iterator(): Iterator<GecTaggerFeaturesOutput> {
        return GecTaggerFeaturesIterator(dataset = dataset, batchSize = batchSize, padId = padId)
    }
}

class GecTaggerFeaturesIterator(
    val dataset: GecTaggerFeaturesData,
    val batchSize: Int, val padId: Int): Iterator<GecTaggerFeaturesOutput>
{

    private var iteratorValue: Int = 0

    override operator fun hasNext(): Boolean{
        return iteratorValue < dataset.size()
    }

    override operator fun next(): GecTaggerFeaturesOutput {

        val subData = hashMapOf<String, List<Any>>()
        val from = iteratorValue
        val to = min(iteratorValue + batchSize, dataset.size())

        val sentences = ArrayList<List<Int>>()
        val offsets = ArrayList<List<Int>>()
        val sentIds = ArrayList<Int>()
        val lens = ArrayList<Int>()

        for (idx in from until to){
            val item = dataset[idx]
            sentences.add(item.sentence)
            offsets.add(item.offset)
            sentIds.add(item.sentId)
            lens.add(item.len)
        }
        iteratorValue = to
        val paddedSentences = padSequences(sentences)
        val paddedOffsets = padSequences(offsets)
        return GecTaggerFeaturesOutput(sentIds = sentIds , sentences = paddedSentences, offsets = paddedOffsets, lens = lens)
    }

    private fun padSequences(sequences: List<List<Int>>) : List<List<Int>>{
        val maxLen = sequences.map { it.size }.maxOrNull()
        val paddedSequences = ArrayList<List<Int>>()
        for (seq in sequences){
            val paddedSentence = mutableListOf<Int>()
            for (item in seq){
                paddedSentence.add(item)
            }
            if (paddedSentence.size < maxLen!!){
                for (i in 0 until maxLen-paddedSentence.size){
                    paddedSentence.add(padId)
                }
            }
            paddedSequences.add(paddedSentence)
        }
        return paddedSequences
    }
}
