package jp.live.ugai.tugraph;

import ai.djl.ndarray.*;
import ai.djl.ndarray.types.*;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.transformer.TransformerEncoderBlock;
import ai.djl.training.ParameterStore;

public class TransformerEncoderExample {

  public static void main(String[] args) throws Exception {
    int batchSize = 2;
    int sequenceLength = 3;
    int numEmbeddings = 10;
    int numHeads = 2;
    int hiddenSize = 8;

    //        DefaultVocabulary.Builder.add("I am a dog.") ;
    NDManager manager = NDManager.newBaseManager();
    NDArray input = manager.ones(new Shape(batchSize, sequenceLength, numEmbeddings));
    TransformerEncoderBlock transformerEncoderBlock =
        new TransformerEncoderBlock(numEmbeddings, numHeads, hiddenSize, 0.5f, Activation::relu);
    Linear linear = Linear.builder().setUnits(2).build();
    Block block = new SequentialBlock().add(transformerEncoderBlock).add(linear);
    block.initialize(
        manager, DataType.FLOAT32, new Shape(batchSize, sequenceLength, numEmbeddings));
    NDList inputs = new NDList(input);
    ParameterStore parameterStore = new ParameterStore(manager, false);
    NDArray output = block.forward(parameterStore, inputs, false).singletonOrThrow();
    System.out.println(output);
  }
}
