package io.kinference.algorithms.gec.tokenizer
import io.kinference.algorithms.gec.tokenizer.utils.whitespaceTokenize

class WordPieceTokenizer(val vocab: Map<String, Int>, val unk_token: String, val max_input_chars_per_word: Int = 100){
    fun tokenize(text: String): List<String>{
        val out = mutableListOf<String>()

        for (token in whitespaceTokenize(text)){
            val chars = token.toCharArray()
            if (chars.size > max_input_chars_per_word){
                out.add(unk_token)
                continue
            }
            var is_bad = false
            var start = 0
            val sub_tokens = mutableListOf<String>()
            while (start < chars.size){
                var end = chars.size
                var cur_substr: String? = null
                while (start < end){
                    var substr = chars.slice(start..end-1).joinToString("")
                    if (start > 0){
                        substr = "##" + substr
                    }
                    if (vocab.containsKey(substr)){
                        cur_substr = substr
                        break
                    }
                    end -= 1
                }
                if (cur_substr == null){
                    is_bad = true
                    break
                }
                sub_tokens.add(cur_substr)
                start = end
            }
            if (is_bad){
                out.add(unk_token)
            }
            else{
                for (item in sub_tokens){
                    out.add(item)
                }
            }
        }
        return out
    }
}
