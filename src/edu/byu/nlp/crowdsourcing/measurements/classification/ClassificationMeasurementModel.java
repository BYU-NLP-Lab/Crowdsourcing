package edu.byu.nlp.crowdsourcing.measurements.classification;

import java.io.PrintWriter;
import java.util.Map;

import com.google.gson.Gson;

import edu.byu.nlp.classify.util.ModelTraining.SupportsTrainingOperations;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.measurements.AbstractMeasurementModelBuilder;
import edu.byu.nlp.crowdsourcing.measurements.AbstractMeasurementModelBuilder.StaticMeasurementModelCounts;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;

public interface ClassificationMeasurementModel extends SupportsTrainingOperations{

  State getCurrentState();
  
  Map<String,Integer> getInstanceIndices();

  double logJoint();

  double[] fitOutOfCorpusInstance(DatasetInstance instance);
  
  
  
  /**
   * Although state values represent variational parameters, 
   * I call them the same thing as the variables whose 
   * distributions they parameterize. This helps avoid 
   * the alphabet soup afflicting MeanFieldMultiAnnState. 
   *
   * @author plf1
   */
  public class State {
    
    private PriorSpecification priors;
    private Map<String,Integer> instanceIndices;
    private double[] nuTheta; // C (one per class)
    private double[][] nuSigma2; // J x 2 (each annotator has an alpha,beta parameters to an inverse gamma) 
    private double[][] logNuY; // N X C (one per instance per class)
    private Dataset data;
    private StaticMeasurementModelCounts staticCounts;

    public State setY(double[][] y){
      this.logNuY=y;
      return this;
    }
    public double[][] getLogNuY(){
      return logNuY;
    }
    
    public State setPriors(PriorSpecification priors){
      this.priors=priors;
      return this;
    }
    public PriorSpecification getPriors(){
      return priors;
    }

    public State setInstanceIndices(Map<String,Integer> instanceIndices){
      this.instanceIndices=instanceIndices;
      return this;
    }
    public Map<String,Integer> getInstanceIndices(){
      return instanceIndices;
    }

    public State setTheta(double[] theta){
      this.nuTheta=theta;
      return this;
    }
    public double[] getTheta(){
      return nuTheta;
    }

    public State setSigma2(double[][] sigma2){
      this.nuSigma2=sigma2;
      return this;
    }
    public double[][] getSigma2(){
      return nuSigma2;
    }

    public State setData(Dataset data){
      this.data=data;
      return this;
    }
    public Dataset getData(){
      return data;
    }

    public State setStaticCounts(StaticMeasurementModelCounts staticCounts) {
      this.staticCounts=staticCounts;
      return this;
    }
    public StaticMeasurementModelCounts getStaticCounts(){
      return staticCounts;
    }

    public int getNumAnnotators(){
      return data.getInfo().getNumAnnotators();
    }
    
    public int getNumClasses(){
      return data.getInfo().getNumClasses();
    }
    
    public int getNumDocuments(){
      return data.getInfo().getNumDocuments();
    }

    public void longDescription(PrintWriter serializeOut){
      serializeOut.print(new Gson().toJson(this));
    }

    public State copy() {
      return new State()
      .setData(data)
      .setInstanceIndices(instanceIndices)
      .setPriors(priors)
      .setSigma2(nuSigma2)
      .setTheta(nuTheta)
      .setY(logNuY)
      ;
    }
  }

}
