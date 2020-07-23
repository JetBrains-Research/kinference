import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.jetbrains.research.kotlin.mpp.inference.misc.pos.POSTest
import org.jetbrains.research.kotlin.mpp.inference.model.Model
import org.jetbrains.research.kotlin.mpp.inference.operators.layer.recurrent.lstm.LSTMLayerTest
import java.io.File

fun main(){
    //Thread.sleep(5000)
    val path = object {}.javaClass.getResource("/pos/test_pos_tagger/").path
    val model = Model.load(path + "model.onnx")
    val dataSet = File(path).list()!!.filter { "test" in it }.map {
        val inputFiles = File("$path/$it").walk().filter { file -> "input" in file.name }

        val inputTensors = inputFiles.map { Utils.getTensor(it) }.toList()
        inputTensors
    }.first()

    val count = 20
    val times = LongArray(count)
    for (i in (0 until count)){
        val startTime = System.currentTimeMillis()
        model.predict(dataSet)
        val endTime = System.currentTimeMillis()
        times[i] = endTime - startTime
    }
    print(times.average())

    //Thread.sleep(5000)
}
