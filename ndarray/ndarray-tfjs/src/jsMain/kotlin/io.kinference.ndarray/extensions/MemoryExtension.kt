package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.ArrayTFJS
import io.kinference.ndarray.core.tidy

fun tidy(fn: () -> Array<ArrayTFJS>): Array<ArrayTFJS> = tidy(fn, null)
