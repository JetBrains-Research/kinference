package io.kinference.ndarray.core

import io.kinference.ndarray.arrays.ArrayTFJS

internal external interface Linalg {
    val qr: (x: ArrayTFJS, fullMatrices: Boolean) -> Array<ArrayTFJS>

    val bandPart: (a: ArrayTFJS, numLower: Int, numUpper: Int) -> ArrayTFJS
}
