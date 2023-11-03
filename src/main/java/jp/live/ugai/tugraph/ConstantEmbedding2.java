package jp.live.ugai.tugraph;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.AbstractIndexedEmbedding;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Optional;

/** An {@link AbstractIndexedEmbedding} that always returns a constant value. */
@SuppressWarnings("rawtypes")
public class ConstantEmbedding2 extends AbstractBlock implements AbstractIndexedEmbedding {

  protected NDArray embedding;

  /**
   * Constructs a constant embedding with the given constant.
   *
   * <p>The constant is assumed to be a fixed value, and starts out as frozen. To unfreeze, use
   * {@link ai.djl.nn.Block#freezeParameters(boolean)}.
   *
   * @param embedding the value to return for all embeddings
   */
  public ConstantEmbedding2(NDArray embedding) {
    this.embedding = embedding;
    freezeParameters(true);
  }

  /** {@inheritDoc} */
  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> params) {
    NDManager manager = inputs.get(0).getManager();
    NDArray base = manager.create(embedding.getShape());
    embedding.copyTo(base);
    System.out.println("Base: " + base);
    Shape shape = inputs.get(0).getShape().addAll(embedding.getShape());
    System.out.println(shape);
    return new NDList(
        base.reshape(1, embedding.size()).repeat(0, inputs.get(0).size()).reshape(shape));
  }

  /** {@inheritDoc} */
  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {inputShapes[0].addAll(embedding.getShape())};
  }

  /** {@inheritDoc} */
  @Override
  public void saveParameters(DataOutputStream os) {
    // Nothing to save
  }

  /** {@inheritDoc} */
  @Override
  public void loadParameters(NDManager manager, DataInputStream is) {
    // Nothing to load
  }

  /** {@inheritDoc} */
  @Override
  public Optional<?> unembed(long index) {
    return Optional.empty();
  }

  /** {@inheritDoc} */
  @Override
  public byte[] encode(Object input) {
    return new byte[0];
  }

  /** {@inheritDoc} */
  @Override
  public Object decode(byte[] byteArray) {
    return null;
  }

  /** {@inheritDoc} */
  @Override
  public long embed(Object item) {
    return 0;
  }

  /** {@inheritDoc} */
  @Override
  public NDArray embed(NDManager manager, Object[] items) {
    NDArray base = manager.create(embedding.getShape());
    embedding.copyTo(base);
    System.out.println("Length: " + items.length);
    return base.repeat(0, items.length)
        .reshape(new Shape(items.length).addAll(embedding.getShape()));
  }

  /** {@inheritDoc} */
  @Override
  public boolean hasItem(Object item) {
    return true;
  }

  public static void main(String[] args) {
    NDManager manager0 = NDManager.newBaseManager();
    System.out.println("Heelo");
    TrainingConfig config =
        new DefaultTrainingConfig(Loss.l2Loss())
            .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

    ConstantEmbedding2 block =
        new ConstantEmbedding2(manager0.create(new float[] {1, 2}, new Shape(2)));
    NDArray base = manager0.create(block.embedding.getShape());
    block.embedding.copyTo(base);
    System.out.println(base.reshape(1, block.embedding.size()).repeat(0, 2));
    Shape shape = new Shape(2).addAll(block.embedding.getShape());
    System.out.println(shape);
    System.out.println(base.reshape(1, block.embedding.size()).repeat(0, 2).reshape(shape));
    block.embed("0");

    try (Model model = Model.newInstance("model")) {
      model.setBlock(block);

      try (Trainer trainer = model.newTrainer(config)) {
        Shape inputShape = new Shape(1, 2);
        trainer.initialize(inputShape);

        NDManager manager = trainer.getManager();
        System.out.println(block.embedding.size());
        System.out.println(block.embed("x"));
        System.out.println(block.embed(new String[] {"x", "y"}));

        System.out.println(trainer.forward(new NDList(manager.create("x"))).singletonOrThrow());

        System.out.println(
            trainer.forward(new NDList(manager.create(new int[] {1, 2}))).singletonOrThrow());
      }
    }
  }
}
