package io.kinference.algorithms.gec.corrector

import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.model.Model
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.LongNDArray

/**
 * Class which hold Bert model for tags prediction
 * @param model name of the model
 */
class Seq2Logits(model: String) {

    private val model = Model.load(model)

    /**
     * Result of tag prediction
     */
    data class LogitResults(val logitsTag: FloatNDArray, val logitsDTags: FloatNDArray)

    operator fun invoke(sentences: LongNDArray, attentionMask: LongNDArray): LogitResults {
        val listResults = model.predict(listOf(sentences.asTensor(name = "input_ids"),
            attentionMask.asTensor(name = "attention_mask"))).map { (it as Tensor).data }

        val logitsTag = listResults[0] as FloatNDArray
        val logitsDTags = listResults[1] as FloatNDArray

        return LogitResults(logitsTag = logitsTag, logitsDTags = logitsDTags)
    }
}
