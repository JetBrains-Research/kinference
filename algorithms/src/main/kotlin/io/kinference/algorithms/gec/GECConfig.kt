package io.kinference.algorithms.gec

import io.kinference.algorithms.gec.corrector.Seq2Logits
import io.kinference.algorithms.gec.encoder.BertTextEncoder
import io.kinference.algorithms.gec.encoder.PreTrainedTextEncoder
import io.kinference.algorithms.gec.preprocessing.TokenVocabulary
import io.kinference.algorithms.gec.preprocessing.VerbsFormVocabulary
import java.io.InputStream

data class GECConfig(
    val model: Seq2Logits,
    val encoder: PreTrainedTextEncoder,
    val labelsVocab: TokenVocabulary,
    val dTagsVocab: TokenVocabulary,
    val verbsVocab: VerbsFormVocabulary,
    val bertTokenizer: BertTextEncoder
) {
    companion object {
        fun loadFrom(loader: (String) -> InputStream): GECConfig {
            fun getBytes(name: String): ByteArray = loader(name).use { it.readBytes() }
            fun getText(name: String): String = loader(name).use { it.reader().readText() }

            val model = Seq2Logits(getBytes("model.onnx"))
            val textProcessor = BertTextEncoder(getText("bert_base_uncased"))
            val labelsVocab = TokenVocabulary.load(getText("labels.txt"))
            val dTagsVocab = TokenVocabulary.load(getText("d_tags.txt"))
            val verbsFormVocab = VerbsFormVocabulary.load(getText("verb_form_vocab.txt"))
            val bertTokenizer = BertTextEncoder(getText("bert_base_uncased"))

            return GECConfig(model, textProcessor, labelsVocab, dTagsVocab, verbsFormVocab, bertTokenizer)
        }
    }

}
