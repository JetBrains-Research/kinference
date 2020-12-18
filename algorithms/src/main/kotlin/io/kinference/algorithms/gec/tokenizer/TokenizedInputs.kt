package io.kinference.algorithms.gec.tokenizer

import io.kinference.ndarray.arrays.IntMap
import io.kinference.ndarray.arrays.IntNDArray

/**
 * Data class which implements inputs for Transformer model
 * Perform inputs for Transformer model with tokenized sentence/sentences [inputsIds] and pad mask [attentionMask]
 *
 * @param inputsIds - inputs for transformer model
 * @param attentionMask - padding mask for transformer model
*/
data class TokenizedInputs(val inputsIds: IntNDArray, val padTokenId: Int) {

    val attentionMask = inputsIds.map(object : IntMap {
        override fun apply(value: Int): Int {
            return if (value != padTokenId)
                1
            else
                0
        }
    })
}
