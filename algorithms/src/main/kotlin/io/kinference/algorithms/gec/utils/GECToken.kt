package io.kinference.algorithms.gec.utils

/**
 * Token-wise correction class
 */
data class TokenCorrection(val tag: String, val errorClass: String, var changedTokens: List<GECToken>)

/**
 * Tokens used in GEC
 */
data class GECToken(val text: String, val encoded: List<Int>, val range: TokenRange,
                    val isFirst: Boolean, val isUsed: Boolean) {

    /**
     * Information about token
     */
    data class TokenRange(val start: Int, val end: Int, val withSpace: Boolean)

    var position: String = "none"

    fun initialTokenPosition(): Int {
        return position.split('.', limit = 2)[0].toInt()
    }

    fun isStartToken(): Boolean {
        return text == "\$START"
    }
}
