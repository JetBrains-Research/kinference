package io.kinference.algorithms.gec.tokenizer

import io.kinference.algorithms.gec.ConfigLoader
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class TokenizerTest {

    private val config = ConfigLoader.v2

    private val tokenizer = config.bertTokenizer //BertTokenizer(vocabPath = Paths.get("/Users/Ivan.Dolgov/ivandolgov/projects/vocabs/bert-base-uncased"))
    private val texts = listOf(
        "Where's your bag?",
        "Horaaaaay!",
        "He is my best friend.",
        "Travelling represents energy , so if you must spend energy, it is more economic and environmentally friendly if you can translate many people with the same energy.",
        "Also, you can use better energy sources and with much better profit with public transport than the personal car (like electricity from solar or wind devices) and then your level of greenhouse gases may be reduced in an important percentage.",
        "As the Lottery would sound more appealing due to the larger winnings , greater public interest and the fact that it is easier to fill in than a pools coupon , fewer people would buy the pools coupon , and this decrease in demand would mean that pools companies such as Littlewoods and Vernon would be forced to make employees redundent due to a decrease in profits .",
        "To conclude, public transport is not useful anymore.",
        "Bus is a transportation which I rarely take.",
        "There is never a lack of work are a means for continuing employment.",
        "Eighteen year olds would not be put in jail for containing alcohol in the car either, which is another high percentage crime among teenagers."
    )

    private val tokenizeAnswer = listOf(
        listOf("where", "'", "s", "your", "bag", "?"),
        listOf("ho", "##ra", "##aa", "##aa", "##y", "!"),
        listOf("he", "is", "my", "best", "friend", "."),
        listOf("travelling", "represents", "energy", ",", "so", "if", "you", "must", "spend", "energy", ",", "it", "is", "more", "economic", "and", "environmentally", "friendly", "if", "you", "can", "translate", "many", "people", "with", "the", "same", "energy", "."),
        listOf("also", ",", "you", "can", "use", "better", "energy", "sources", "and", "with", "much", "better", "profit", "with", "public", "transport", "than", "the", "personal", "car", "(", "like", "electricity", "from", "solar", "or", "wind", "devices", ")", "and", "then", "your", "level", "of", "greenhouse", "gases", "may", "be", "reduced", "in", "an", "important", "percentage", "."),
        listOf("as", "the", "lottery", "would", "sound", "more", "appealing", "due", "to", "the", "larger", "winning", "##s", ",", "greater", "public", "interest", "and", "the", "fact", "that", "it", "is", "easier", "to", "fill", "in", "than", "a", "pools", "coup", "##on", ",", "fewer", "people", "would", "buy", "the", "pools", "coup", "##on", ",", "and", "this", "decrease", "in", "demand", "would", "mean", "that", "pools", "companies", "such", "as", "little", "##woods", "and", "vernon", "would", "be", "forced", "to", "make", "employees", "red", "##und", "##ent", "due", "to", "a", "decrease", "in", "profits", "."),
        listOf("to", "conclude", ",", "public", "transport", "is", "not", "useful", "anymore", "."),
        listOf("bus", "is", "a", "transportation", "which", "i", "rarely", "take", "."),
        listOf("there", "is", "never", "a", "lack", "of", "work", "are", "a", "means", "for", "continuing", "employment", "."),
        listOf("eighteen", "year", "olds", "would", "not", "be", "put", "in", "jail", "for", "containing", "alcohol", "in", "the", "car", "either", ",", "which", "is", "another", "high", "percentage", "crime", "among", "teenagers", ".")
    )

    private val encodedAnswer = listOf(
        listOf(101, 2073, 1005, 1055, 2115, 4524, 1029, 102),
        listOf(101, 7570, 2527, 11057, 11057, 2100, 999, 102),
        listOf(101, 2002, 2003, 2026, 2190, 2767, 1012, 102),
        listOf(101, 8932, 5836, 2943, 1010, 2061, 2065, 2017, 2442, 5247, 2943, 1010, 2009, 2003, 2062, 3171, 1998, 25262, 5379, 2065, 2017, 2064, 17637, 2116, 2111, 2007, 1996, 2168, 2943, 1012, 102),
        listOf(101, 2036, 1010, 2017, 2064, 2224, 2488, 2943, 4216, 1998, 2007, 2172, 2488, 5618, 2007, 2270, 3665, 2084, 1996, 3167, 2482, 1006, 2066, 6451, 2013, 5943, 2030, 3612, 5733, 1007, 1998, 2059, 2115, 2504, 1997, 16635, 15865, 2089, 2022, 4359, 1999, 2019, 2590, 7017, 1012, 102),
        listOf(101, 2004, 1996, 15213, 2052, 2614, 2062, 16004, 2349, 2000, 1996, 3469, 3045, 2015, 1010, 3618, 2270, 3037, 1998, 1996, 2755, 2008, 2009, 2003, 6082, 2000, 6039, 1999, 2084, 1037, 12679, 8648, 2239, 1010, 8491, 2111, 2052, 4965, 1996, 12679, 8648, 2239, 1010, 1998, 2023, 9885, 1999, 5157, 2052, 2812, 2008, 12679, 3316, 2107, 2004, 2210, 25046, 1998, 11447, 2052, 2022, 3140, 2000, 2191, 5126, 2417, 8630, 4765, 2349, 2000, 1037, 9885, 1999, 11372, 1012, 102),
        listOf(101, 2000, 16519, 1010, 2270, 3665, 2003, 2025, 6179, 4902, 1012, 102),
        listOf(101, 3902, 2003, 1037, 5193, 2029, 1045, 6524, 2202, 1012, 102),
        listOf(101, 2045, 2003, 2196, 1037, 3768, 1997, 2147, 2024, 1037, 2965, 2005, 5719, 6107, 1012, 102),
        listOf(101, 7763, 2095, 19457, 2052, 2025, 2022, 2404, 1999, 7173, 2005, 4820, 6544, 1999, 1996, 2482, 2593, 1010, 2029, 2003, 2178, 2152, 7017, 4126, 2426, 12908, 1012, 102)

    )

//    private val encoded = listOf(101, 2036, 1010, 2017, 2064, 2224, 2488, 2943, 4216, 1998, 2007, 2172, 2488, 5618, 2007, 2270, 3665, 2084, 1996, 3167, 2482, 1006, 2066, 6451, 2013, 5943, 2030, 3612, 5733, 1007, 1998, 2059, 2115, 2504, 1997, 16635, 15865, 2089, 2022, 4359, 1999, 2019, 2590, 7017, 1012, 102)
//    private val inputIds = IntNDArray(shape = IntArray(size = 2, init = { i: Int ->  if (i==0) 1 else encoded.size}), init = { i: Int -> encoded[i]})
//    private val tokenizedInputs = TokenizedInputs(inputsIds = inputIds, padTokenId = 0)
//
//    private val encodedAll
//    private val inputsIdAll
//    private val tokenizedInputsAll

    @Test
    @Tag("heavy")
    fun testTokenize(){
        for ((idx, text) in texts.withIndex()){
            val res = tokenizer.encodeAsTokens(text)
            assertEquals(res, tokenizeAnswer[idx])
        }
    }

    @Test
    @Tag("heavy")
    fun testEncode(){
        for ((idx, text) in texts.withIndex()){
            val res = tokenizer.encodeAsIds(text, withSpecialTokens = true)
            assertEquals(res, encodedAnswer[idx])
        }
    }

    @Test
    @Tag("heavy")
    fun testEncodeBatch(){
        val res = tokenizer.encodeAsIds(texts, withSpecialTokens = true)
        assertEquals(res, encodedAnswer)
    }

//    @Test
//    @Tag("heavy")
//    fun testEncodeIntNDArray(){
//        val arr = tokenizer.encode2tensor(texts[4], addSpecialTokens = true)
//        assertTensor(arr.inputsIds.asTensor(), tokenizedInputs.inputsIds.asTensor())
//        assertTensor(arr.attentionMask.asTensor(), tokenizedInputs.attentionMask.asTensor())
//    }
//
//    @Test
//    @Tag("heavy")
//    fun testEncodeIntNDArrayBatch(){
//        val arr = tokenizer.encodeBatch2tensor(texts, addSpecialTokens = true)
//        assertTensor(arr.inputsIds.asTensor(),)
//        assertTensor(arr.attentionMask.asTensor())
//    }
}
