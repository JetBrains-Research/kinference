package io.kinference.algorithms.gec

import io.kinference.algorithms.gec.corrector.Seq2Logits
import io.kinference.algorithms.gec.preprocessing.*
import io.kinference.algorithms.gec.encoder.BertTextEncoder
import io.kinference.loaders.S3Client
import java.io.File
import java.nio.file.Path

object ConfigLoader {
    private val testData = File("../build/test-data")

    private const val V2Path = "/bert/gec/en/standard/v2/"

    val v2: Config by lazy { loadConfigs(V2Path, "tests$V2Path") }

    private fun loadConfigs(path: String, prefix: String): Config {
        val toFolder = File(testData, path)
        S3Client.copyObjects(prefix, toFolder)

        return Config.loadFromFolder(toFolder)
    }
}

data class Config(
    val model: Seq2Logits,
    val textProcessor: TransformersTextProcessor,
    val labelsVocab: Vocabulary,
    val dTagsVocab: Vocabulary,
    val verbsVocab: VerbsFormVocabulary,
    val bertTokenizer: BertTextEncoder
) {
    companion object {
        fun loadFromFolder(folder: File): Config {

            fun getPathInFolder(fileName: String): String {
                val file = File(folder, fileName)

                return if (file.exists()) file.path else error("File $fileName not found")
            }

            val model = Seq2Logits(getPathInFolder("model.onnx"))
            val textProcessor = TransformersTextProcessor(getPathInFolder("bert_base_uncased"))
            val labelsVocab = Vocabulary.loadFromFile(getPathInFolder("labels.txt"))
            val dTagsVocab = Vocabulary.loadFromFile(getPathInFolder("d_tags.txt"))
            val verbsFormVocab = VerbsFormVocabulary.setupVerbsFormVocab(getPathInFolder("verb_form_vocab.txt"))
            val bertTokenizer = BertTextEncoder(File(getPathInFolder("bert_base_uncased")))

            return Config(model, textProcessor, labelsVocab, dTagsVocab, verbsFormVocab, bertTokenizer)
        }
    }

}
