package io.kinference.algorithms.gec.tokenizer

/** Defines range in text in which token is located and if it had space on the left of it. */
data class TokenRange(val start: Int, val endExclusive: Int, val withSpace: Boolean) {
    val range = IntRange(start, endExclusive - 1)

    companion object {
        /**
         * Finds tokens from [tokens] in [text].
         *
         * Note, that [tokens] must be in the same order they would be in text
         * and all should be presented in it.
         *
         * Note, that function would look for the first match for each token.
         *
         * Basically, you should use it following way:
         *
         * text: "It it is a good day -- really good"
         * tokens: ["Tt", "it", "good", "really", "good"]
         */
        fun findTokensInText(text: String, tokens: List<String>, textWithSpace: Boolean = false): List<TokenRange> {
            val result = ArrayList<TokenRange>()
            var startFrom = 0
            for ((idx, token) in tokens.withIndex()) {
                val startIdxAndString = text.indexOf(token, startIndex = startFrom)

                require(startIdxAndString != -1)
                val withSpace: Boolean = if (idx == 0 && textWithSpace) {
                    true
                } else {
                    startIdxAndString >= startFrom + 1
                }
                result.add(
                    TokenRange(
                        start = startIdxAndString,
                        endExclusive = startIdxAndString + token.length,
                        withSpace = withSpace
                    )
                )
                startFrom = startIdxAndString + token.length

            }
            return result
        }

    }
}
