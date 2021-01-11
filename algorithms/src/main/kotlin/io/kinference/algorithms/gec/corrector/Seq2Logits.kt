package io.kinference.algorithms.gec.corrector

import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.model.Model
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.LongNDArray

/**
 * Result of tag prediction
 */
data class LogitResults(val logitsTag: FloatNDArray, val logitsDTags: FloatNDArray)

/**
 * Class which hold Bert model for tags prediction
 */
class Seq2Logits(modelPath: String) {

    private val model = Model.load(modelPath)

    operator fun invoke(sents: LongNDArray, attentionMask: LongNDArray): LogitResults {
        val listResults = model.predict(listOf(sents.asTensor(name = "input_ids"),
            attentionMask.asTensor(name = "attention_mask"))).map { (it as Tensor).data }

        val logitsTag = listResults[0] as FloatNDArray
        val logitsDTags = listResults[1] as FloatNDArray

        return LogitResults(logitsTag = logitsTag, logitsDTags = logitsDTags)
    }
}
