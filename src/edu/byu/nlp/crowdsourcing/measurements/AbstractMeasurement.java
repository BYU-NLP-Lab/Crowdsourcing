package edu.byu.nlp.crowdsourcing.measurements;

import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.Measurement;

public abstract class AbstractMeasurement<L> implements Measurement<L> {

  private Dataset dataset;
  private int annotator;

  public AbstractMeasurement(Dataset dataset, int annotator){
    this.dataset=dataset;
    this.annotator=annotator;
  }
  
  @Override
  public Dataset getDataset() {
    return dataset;
  }

  @Override
  public int getAnnotator() {
    return annotator;
  }
  
}
