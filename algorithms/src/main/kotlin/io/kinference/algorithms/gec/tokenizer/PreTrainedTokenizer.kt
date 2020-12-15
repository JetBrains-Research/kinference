package io.kinference.algorithms.gec.tokenizer

interface PreTrainedTokenizer{
    val vocabSize: Int
    val doLowerCase: Boolean

    fun length() : Int{
        return vocabSize
    }

   fun tokenize_(text: String): List<String>

   fun convertTokenToId_(token: String): Int

   fun convertIdToToken_(id: Int): String

    fun splitOnToken(tok: String, text: String): List<String>{
        val result = mutableListOf<String>()
        val split_text = text.split(tok)

        for ((i, subText) in split_text.withIndex()){
            var mSubText: String = subText
            if (i < split_text.size - 1){
                mSubText = mSubText.trimEnd()
            }
            if (i > 0){
                mSubText = mSubText.trimStart()
            }
            if (i == 0 && !mSubText.isNullOrEmpty()){
                result.add(tok)
            }
            else if(i == mSubText.length - 1){
                if (!mSubText.isNullOrEmpty()){
                    result.add(mSubText)
                }
            }
            else{
                if (!mSubText.isNullOrEmpty()){
                    result.add(mSubText)
                }
                result.add(tok)
            }
        }
        return result
    }

    fun splitOnTokens(tok_list: List<String>, text: String): List<String>{
        if (text.trimEnd().trimStart().isNullOrEmpty()){
            return listOf()
        }
        if (tok_list.isNullOrEmpty()){
            return tokenize_(text)
        }
        val tokenized_text = mutableListOf<String>()
        var text_list = mutableListOf(text)

        for (tok in tok_list){
            val token_text = mutableListOf<String>()
            for (sub_text in text_list){
                token_text += splitOnToken(tok, sub_text)
            }
            text_list = token_text
        }

        val result = mutableListOf<List<String>>()

        for (token in tokenized_text){
            result.add(tokenize_(token))
        }
        return result.flatten()

    }

    fun tokenize(text: String): List<String>{
        val mText: String = if (doLowerCase){
            text.toLowerCase()
        }
        else{
            text
        }
        return splitOnTokens(listOf(), mText)
    }

    fun convertTokensToIds(tokens: List<String>): List<Int> {
        if (tokens.isNullOrEmpty()){
            return listOf<Int>()
        }

        val ids = mutableListOf<Int>()
        for (token in tokens){
            ids.add(convertTokenToId_(token))
        }
        return ids
    }

    fun convertIdsToTokens(ids: List<Int>): List<String>{
        val tokens = mutableListOf<String>()
        for (index in ids){
            tokens.add(convertIdToToken_(index))
        }
        return tokens
    }

    fun covertTokensToString(tokens: List<String>): String{
        return tokens.joinToString(" ")
    }
}
