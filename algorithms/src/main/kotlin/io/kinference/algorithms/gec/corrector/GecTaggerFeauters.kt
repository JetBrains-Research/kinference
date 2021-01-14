package io.kinference.algorithms.gec.corrector

import kotlin.math.min

/**
 * GecTagger feature from sentence
 * @param sent - string which represents a sentence
 * @param sentId - Id of sentence
 * @param tokSent - sentence which splited on tokens
 * @param encTokSent - encoded tokens from tokenizer
 * @param flatEncSent - flatted tokens from encTokSent
 * @param offset - indexes from flatEncSent which represent where token to start
 */
data class GecTaggerFeatures(val sent: String,
                             val sentId: Int,
                             val tokSent: List<String>,
                             val encTokSent: List<List<Int>>,
                             val flatEncSent: List<Int>,
                             val offset: List<Int>){

    /**
     * Batch Output from DataLoader
     */
    data class Output(val sentIds: List<Int>,
                                       val sentences: List<List<Int>>,
                                       val offsets: List<List<Int>>,
                                       val lens: List<Int>)

    /**
     * Iterator output
     */
    data class IteratorOut(val sentId: Int,
                                    val sentence: List<Int>,
                                    val offset: List<Int>,
                                    val len: Int)

    /**
     * Features data storage class
     */
    class Data(val data: List<GecTaggerFeatures>) {
        operator fun get(index: Int): IteratorOut {
            val sentence = data[index]
            return IteratorOut(
                sentence.sentId,
                sentence.flatEncSent,
                sentence.offset,
                sentence.tokSent.size)
        }

        fun size(): Int {
            return data.size
        }
    }

    /**
     * DataLoader implementation for specific GEC task
     */
    class DataLoader(
        val dataset: Data,
        val batchSize: Int, val padId: Int) : Iterable<Output> {

        override fun iterator(): Iterator<Output> {
            return FeaturesIterator(dataset = dataset, batchSize = batchSize, padId = padId)
        }
    }

    /**
     * Helper class for DataLoader
     */
    class FeaturesIterator(
        val dataset: Data,
        val batchSize: Int, val padId: Int) : Iterator<Output> {

        private var iteratorValue: Int = 0

        override operator fun hasNext(): Boolean {
            return iteratorValue < dataset.size()
        }

        override operator fun next(): Output {

            if (iteratorValue >= dataset.size()){
                throw NoSuchElementException()
            }

            val from = iteratorValue
            val to = min(iteratorValue + batchSize, dataset.size())

            val sentences = ArrayList<List<Int>>()
            val offsets = ArrayList<List<Int>>()
            val sentIds = ArrayList<Int>()
            val lens = ArrayList<Int>()

            for (idx in from until to) {
                val item = dataset[idx]
                sentences.add(item.sentence)
                offsets.add(item.offset)
                sentIds.add(item.sentId)
                lens.add(item.len)
            }
            iteratorValue = to
            val paddedSentences = padSequences(sentences)
            val paddedOffsets = padSequences(offsets)
            return Output(sentIds = sentIds, sentences = paddedSentences, offsets = paddedOffsets, lens = lens)
        }

        private fun padSequences(sequences: List<List<Int>>): List<List<Int>> {
            val maxLen = sequences.map { it.size }.maxOrNull()
            val paddedSequences = ArrayList<List<Int>>()
            for (seq in sequences) {
                val paddedSentence = ArrayList<Int>()
                for (item in seq) {
                    paddedSentence.add(item)
                }
                if (paddedSentence.size < maxLen!!) {
                    for (i in 0 until maxLen - paddedSentence.size) {
                        paddedSentence.add(padId)
                    }
                }
                paddedSequences.add(paddedSentence)
            }
            return paddedSequences
        }
    }
}
