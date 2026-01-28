package jp.live.ugai.tugraph

import ai.djl.Application
import ai.djl.engine.Engine
import ai.djl.huggingface.translator.FillMaskTranslatorFactory
import ai.djl.modality.Classifications
import ai.djl.repository.zoo.Criteria
import ai.djl.repository.zoo.ZooModel
import ai.djl.training.util.ProgressBar

/** Runs a fill-mask example using a HuggingFace model. */
fun main() {
    val sentence = "I have watched this [MASK] and it was awesome."

    val criteria =
        Criteria.builder()
            .optApplication(Application.NLP.FILL_MASK)
            .setTypes(String::class.java, Classifications::class.java)
            .optTranslatorFactory(FillMaskTranslatorFactory())
            .optArgument("topK", 5)
            .optEngine(Engine.getDefaultEngineName())
            .optProgress(ProgressBar())
            .build()

    val model: ZooModel<String, Classifications> = criteria.loadModel()
    model.newPredictor().use { predictor ->
        val result = predictor.predict(sentence)
        println(result.topK<Classifications.Classification>(5))
    }
}

/** Marker class for the fill-mask example. */
class BertExample2
