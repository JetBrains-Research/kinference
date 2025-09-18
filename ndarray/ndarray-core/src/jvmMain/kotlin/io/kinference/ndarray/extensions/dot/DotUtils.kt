package io.kinference.ndarray.extensions.dot

object DotUtils {
    const val CHUNK_SIZE = 64
    var MIN_DATA_PER_LAUNCH = 262100
    const val PAGE_BYTES = 4 * 1024
}
