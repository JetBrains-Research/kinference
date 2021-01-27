package io.kinference.algorithms.gec.corrector

/**
 * GecTagger feature from sentence
 * @param sent string which represents a sentence
 * @param sentId Id of sentence
 * @param tokSent sentence which split on tokens
 * @param encTokSent encoded tokens from tokenizer
 * @param flatEncSent flatted tokens from encTokSent
 * @param offset indexes from flatEncSent which represent where token to start
 */
data class GECTaggerFeatures(
    val sent: String,
    val sentId: Int,
    val tokSent: List<String>,
    val encTokSent: List<List<Int>>,
    val flatEncSent: List<Int>,
    val offset: List<Int>
) {

    /**
     * DataLoader implementation for specific GEC task.
     *
     * Helps preparing data and aggregating it into batches
     */
    class DataLoader(val dataset: List<GECTaggerFeatures>, val batchSize: Int, val padId: Int) : Iterable<DataLoader.Batch> {
        /** Batch Output from DataLoader */
        data class Batch(
            val sentIds: List<Int>,
            val sentences: List<List<Int>>,
            val offsets: List<List<Int>>,
            val lens: List<Int>
        )

        private val batched = dataset.chunked(batchSize).map { out ->
            Batch(
                out.map { it.sentId },
                padSequences(out.map { it.flatEncSent }),
                padSequences(out.map { it.offset }),
                out.map { it.tokSent.size }
            )
        }

        override fun iterator(): Iterator<Batch> = batched.iterator()

        private fun padSequences(sequences: List<List<Int>>): List<List<Int>> {
            val maxLen = sequences.map<List<Int>, Int> { it.size }.max()!!
            val paddedSequences = ArrayList<List<Int>>()
            for (seq in sequences) {
                val paddedSentence = ArrayList<Int>()
                for (item in seq) {
                    paddedSentence.add(item)
                }
                if (paddedSentence.size < maxLen) {
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
