package io.kinference.algorithms.gec.utils

/**
 * Token-wise correction class
 */
data class TokenCorrection(val tag: String, val errorClass: String, var changedTokens: List<Token>)

/**
 * Token class
 */
data class Token(var text: String, var encodedData: List<Int>,
                 val tokenRange: TokenRange, val isFirst: Boolean, var isUsed: Boolean) {

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
