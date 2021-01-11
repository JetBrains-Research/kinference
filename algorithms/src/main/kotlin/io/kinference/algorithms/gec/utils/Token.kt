package io.kinference.algorithms.gec.utils

data class TokenRange(val start: Int, val end: Int, val withSpace: Boolean)

data class TokenCorrection(val tag: String, val errorClass: String, var changedTokens: List<Token>)

data class Token(var text: String, var encodedData: List<Int>,
                 val tokenRange: TokenRange, val isFirst: Boolean, var isUsed: Boolean) {

    var position: String = "none"

    fun initialTokenPosition(): Int {
        return position.split('.', limit = 2)[0].toInt()
    }

    fun isStartToken(): Boolean {
        return text == "\$START"
    }

}
